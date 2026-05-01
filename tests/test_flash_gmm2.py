import unittest

import torch

from flash_gmm2 import (
    FlashGMM,
    batch_gmm_Diagonal,
    batch_gmm_Full,
    batch_gmm_Spherical,
    batch_gmm_Tied,
    fused_single_tile_update_config,
    triton_fused_single_tile_update_diag,
    triton_fused_single_tile_update_spherical,
    triton_fused_single_tile_update_tied_native,
)
from flash_gmm2._runtime import triton_spherical_supported
from flash_gmm2.large_n import (
    large_n_predict_cpu,
    large_n_predict_proba_cpu,
    large_n_score_samples_cpu,
)
from flash_gmm2.torch_fallback import (
    _compute_chunk_logits,
    _compute_diag_chunk_logits,
    _compute_tied_chunk_logits,
    _precision_and_logdet,
    diagonal_assign_torch_native_chunked,
    diagonal_predict_proba_torch_native_chunked,
    diagonal_score_samples_torch_native_chunked,
    tied_predict_proba_torch_native_chunked,
    tied_score_samples_torch_native_chunked,
)
from gmmxx import GMMXX


class FlashGMMTests(unittest.TestCase):
    def test_runtime_gate_policy(self):
        self.assertFalse(triton_spherical_supported(0, 64))
        self.assertFalse(triton_spherical_supported(128, 0))
        self.assertFalse(triton_spherical_supported(256, 64))
        self.assertFalse(triton_spherical_supported(128, 4096))
        self.assertTrue(triton_spherical_supported(128, 2048))

    def test_batch_api_shapes(self):
        torch.manual_seed(0)
        x0 = torch.randn(1, 128, 2) * 0.2 + torch.tensor([[[0.0, 0.0]]])
        x1 = torch.randn(1, 128, 2) * 0.3 + torch.tensor([[[4.0, 4.0]]])
        x = torch.cat([x0, x1], dim=1)

        labels, means, variances, weights, info = batch_gmm_Spherical(
            x,
            n_components=2,
            max_iters=20,
            tol=1e-5,
            init_params="random",
            chunk_size_N=64,
            chunk_size_K=2,
        )

        self.assertEqual(labels.shape, (1, 256))
        self.assertEqual(means.shape, (1, 2, 2))
        self.assertEqual(variances.shape, (1, 2))
        self.assertEqual(weights.shape, (1, 2))
        self.assertGreaterEqual(int(info["n_iter"]), 1)
        self.assertTrue(torch.isfinite(means).all())
        self.assertTrue(torch.isfinite(variances).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-4))

    def test_class_predict_proba_and_score(self):
        torch.manual_seed(1)
        x = torch.cat(
            [
                torch.randn(96, 3) * 0.15 + torch.tensor([0.0, 0.0, 0.0]),
                torch.randn(96, 3) * 0.20 + torch.tensor([3.0, 3.0, 3.0]),
            ],
            dim=0,
        )

        model = FlashGMM(
            d=3,
            k=2,
            niter=25,
            tol=1e-5,
            init_params="random",
            chunk_size_data=64,
            chunk_size_centroids=2,
        )
        labels = model.fit_predict(x)
        pred = model.predict(x[:32])
        probs = model.predict_proba(x[:32])
        scores = model.score_samples(x[:32])

        self.assertEqual(labels.shape, (192,))
        self.assertEqual(pred.shape, (32,))
        self.assertEqual(probs.shape, (32, 2))
        self.assertEqual(scores.shape, (32,))
        self.assertTrue(
            torch.allclose(
                probs.sum(dim=-1),
                torch.ones(32, device=probs.device, dtype=probs.dtype),
                atol=1e-4,
            )
        )
        self.assertTrue(torch.isfinite(scores).all())

    def test_sklearn_style_api_and_learned_attributes(self):
        torch.manual_seed(10)
        x = torch.cat(
            [
                torch.randn(64, 3) * 0.2,
                torch.randn(64, 3) * 0.3 + 3.0,
            ],
            dim=0,
        )

        model = FlashGMM(
            n_components=2,
            max_iter=4,
            random_state=10,
            init_params="random",
            covariance_type="diagonal",
            compute_labels_on_fit=False,
        )
        self.assertIsNone(model.d)
        model.fit(x)

        self.assertEqual(model.d, 3)
        self.assertEqual(model.k, 2)
        self.assertEqual(model.niter, 4)
        self.assertEqual(model.covariance_type, "diag")
        self.assertIsNone(model.labels_)
        self.assertEqual(model.means_.shape, (2, 3))
        self.assertEqual(model.weights_.shape, (2,))
        self.assertEqual(model.covariances_.shape, (2, 3))
        self.assertTrue(torch.isfinite(torch.tensor(model.bic(x))))
        self.assertTrue(torch.isfinite(torch.tensor(model.aic(x))))

        params = model.get_params()
        self.assertEqual(params["n_components"], 2)
        self.assertEqual(params["max_iter"], 4)
        model.set_params(n_components=3, max_iter=5, covariance_type="spherical")
        self.assertEqual(model.k, 3)
        self.assertEqual(model.niter, 5)
        self.assertEqual(model.covariance_type, "spherical")
        self.assertIsNone(model.means_)

    def test_gmmxx_public_alias(self):
        self.assertIs(GMMXX, FlashGMM)

    def test_approx_topk_em_all_covariance_shapes(self):
        torch.manual_seed(11)
        centers = torch.tensor(
            [
                [-3.0, -3.0, 0.0],
                [3.0, -3.0, 1.0],
                [-3.0, 3.0, -1.0],
                [3.0, 3.0, 0.5],
            ]
        )
        x = torch.cat(
            [torch.randn(48, 3) * 0.25 + center for center in centers],
            dim=0,
        )

        for covariance_type in ["spherical", "diag", "tied", "full"]:
            model = FlashGMM(
                d=3,
                k=4,
                niter=3,
                tol=0.0,
                init_params="random",
                reg_covar=1e-4,
                use_triton=False,
                chunk_size_data=64,
                chunk_size_centroids=2,
                covariance_type=covariance_type,
                compute_labels_on_fit=False,
                approx_top_k=2,
            )
            model.fit(x)
            probs = model.predict_proba(x[:16])

            self.assertTrue(model.approximate_em_enabled_)
            self.assertEqual(model.approx_top_k_, 2)
            self.assertFalse(model.triton_streaming_update_enabled_)
            self.assertIsNone(model.cluster_ids_b)
            self.assertEqual(probs.shape, (16, 4))
            self.assertTrue(torch.isfinite(model.means_b).all())
            self.assertTrue(torch.isfinite(model.variances_b).all())
            self.assertTrue(torch.isfinite(model.weights_b).all())
            self.assertTrue(
                torch.allclose(
                    probs.sum(dim=-1),
                    torch.ones(16, device=probs.device, dtype=probs.dtype),
                    atol=1e-4,
                )
            )

    def test_invalid_approx_topk_rejected(self):
        with self.assertRaises(ValueError):
            FlashGMM(d=2, k=2, approx_top_k=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton approximate top-k EM")
    def test_spherical_approx_topk_triton_matches_torch(self):
        torch.manual_seed(12)
        x = torch.randn(2048, 16, device="cuda")

        common_kwargs = {
            "d": 16,
            "k": 64,
            "niter": 1,
            "tol": 0.0,
            "seed": 12,
            "init_params": "random",
            "chunk_size_data": 1024,
            "chunk_size_centroids": 64,
            "covariance_type": "spherical",
            "compute_labels_on_fit": False,
            "approx_top_k": 8,
            "device": torch.device("cuda"),
        }
        triton_model = FlashGMM(use_triton=True, **common_kwargs).fit(x)
        torch_model = FlashGMM(use_triton=False, **common_kwargs).fit(x)

        self.assertTrue(triton_model.triton_approx_topk_enabled_)
        self.assertFalse(torch_model.triton_approx_topk_enabled_)
        self.assertTrue(torch.allclose(triton_model.means_b, torch_model.means_b, atol=5e-4, rtol=5e-4))
        self.assertTrue(torch.allclose(triton_model.variances_b, torch_model.variances_b, atol=5e-4, rtol=5e-4))
        self.assertTrue(torch.allclose(triton_model.weights_b, torch_model.weights_b, atol=5e-4, rtol=5e-4))
        self.assertAlmostEqual(triton_model.lower_bound_, torch_model.lower_bound_, places=4)

    def test_diagonal_batch_api_shapes(self):
        torch.manual_seed(2)
        x0 = torch.randn(1, 96, 3) * torch.tensor([[[0.1, 0.4, 0.7]]])
        x1 = torch.randn(1, 96, 3) * torch.tensor([[[0.8, 0.2, 0.3]]]) + torch.tensor([[[3.0, 3.0, 3.0]]])
        x = torch.cat([x0, x1], dim=1)

        labels, means, variances, weights, info = batch_gmm_Diagonal(
            x,
            n_components=2,
            max_iters=10,
            tol=1e-5,
            init_params="random",
            chunk_size_N=64,
            chunk_size_K=2,
        )

        self.assertEqual(labels.shape, (1, 192))
        self.assertEqual(means.shape, (1, 2, 3))
        self.assertEqual(variances.shape, (1, 2, 3))
        self.assertEqual(weights.shape, (1, 2))
        self.assertGreaterEqual(int(info["n_iter"]), 1)
        self.assertTrue(torch.isfinite(means).all())
        self.assertTrue(torch.isfinite(variances).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(variances > 0))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-4))

    def test_diagonal_class_predict_proba_and_score(self):
        torch.manual_seed(3)
        x = torch.cat(
            [
                torch.randn(80, 4) * torch.tensor([0.1, 0.2, 0.5, 0.9]),
                torch.randn(80, 4) * torch.tensor([0.8, 0.3, 0.2, 0.1])
                + torch.tensor([3.0, 2.0, 4.0, 1.0]),
            ],
            dim=0,
        )

        model = FlashGMM(
            d=4,
            k=2,
            niter=12,
            tol=1e-5,
            init_params="random",
            chunk_size_data=64,
            chunk_size_centroids=2,
            covariance_type="diag",
        )
        labels = model.fit_predict(x)
        pred = model.predict(x[:32])
        probs = model.predict_proba(x[:32])
        scores = model.score_samples(x[:32])

        self.assertEqual(labels.shape, (160,))
        self.assertEqual(pred.shape, (32,))
        self.assertEqual(probs.shape, (32, 2))
        self.assertEqual(scores.shape, (32,))
        self.assertEqual(model.variances_b.shape, (1, 2, 4))
        self.assertFalse(model.triton_estep_enabled_)
        self.assertTrue(
            torch.allclose(
                probs.sum(dim=-1),
                torch.ones(32, device=probs.device, dtype=probs.dtype),
                atol=1e-4,
            )
        )
        self.assertTrue(torch.isfinite(scores).all())

    def test_full_and_tied_batch_api_shapes(self):
        torch.manual_seed(4)
        x = torch.cat(
            [
                torch.randn(1, 64, 3) @ torch.tensor([[[0.6, 0.2, 0.0], [0.1, 0.4, 0.1], [0.0, 0.3, 0.5]]]).squeeze(0),
                torch.randn(1, 64, 3) @ torch.tensor([[[0.3, 0.0, 0.2], [0.2, 0.7, 0.1], [0.1, 0.1, 0.4]]]).squeeze(0)
                + torch.tensor([[[3.0, 2.0, 4.0]]]),
            ],
            dim=1,
        )

        full_labels, full_means, full_covs, full_weights, _ = batch_gmm_Full(
            x,
            n_components=2,
            max_iters=4,
            tol=0.0,
            init_params="random",
            chunk_size_N=32,
            chunk_size_K=2,
        )
        tied_labels, tied_means, tied_cov, tied_weights, _ = batch_gmm_Tied(
            x,
            n_components=2,
            max_iters=4,
            tol=0.0,
            init_params="random",
            chunk_size_N=32,
            chunk_size_K=2,
        )

        self.assertEqual(full_labels.shape, (1, 128))
        self.assertEqual(full_means.shape, (1, 2, 3))
        self.assertEqual(full_covs.shape, (1, 2, 3, 3))
        self.assertEqual(full_weights.shape, (1, 2))
        self.assertEqual(tied_labels.shape, (1, 128))
        self.assertEqual(tied_means.shape, (1, 2, 3))
        self.assertEqual(tied_cov.shape, (1, 3, 3))
        self.assertEqual(tied_weights.shape, (1, 2))
        self.assertTrue(torch.isfinite(full_covs).all())
        self.assertTrue(torch.isfinite(tied_cov).all())
        self.assertTrue(torch.allclose(full_weights.sum(dim=-1), torch.ones(1), atol=1e-4))
        self.assertTrue(torch.allclose(tied_weights.sum(dim=-1), torch.ones(1), atol=1e-4))

    def test_full_and_tied_class_predict_proba_and_score(self):
        torch.manual_seed(5)
        x = torch.cat(
            [
                torch.randn(72, 3) @ torch.tensor([[0.5, 0.2, 0.0], [0.0, 0.3, 0.2], [0.2, 0.1, 0.4]]),
                torch.randn(72, 3) @ torch.tensor([[0.4, 0.0, 0.1], [0.2, 0.5, 0.1], [0.0, 0.2, 0.3]])
                + torch.tensor([3.0, 3.0, 2.0]),
            ],
            dim=0,
        )

        for covariance_type, expected_cov_shape in [("tied", (1, 3, 3)), ("full", (1, 2, 3, 3))]:
            model = FlashGMM(
                d=3,
                k=2,
                niter=4,
                tol=0.0,
                init_params="random",
                chunk_size_data=48,
                chunk_size_centroids=2,
                covariance_type=covariance_type,
            )
            labels = model.fit_predict(x)
            probs = model.predict_proba(x[:24])
            scores = model.score_samples(x[:24])

            self.assertEqual(labels.shape, (144,))
            self.assertEqual(probs.shape, (24, 2))
            self.assertEqual(scores.shape, (24,))
            self.assertEqual(model.variances_b.shape, expected_cov_shape)
            self.assertFalse(model.triton_estep_enabled_)
            self.assertTrue(
                torch.allclose(
                    probs.sum(dim=-1),
                    torch.ones(24, device=probs.device, dtype=probs.dtype),
                    atol=1e-4,
                )
            )
            self.assertTrue(torch.isfinite(scores).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CPU large-N streaming")
    def test_large_cpu_streaming_all_covariance_modes(self):
        torch.manual_seed(6)
        x = torch.cat(
            [
                torch.randn(128, 3) @ torch.tensor([[0.45, 0.12, 0.00], [0.05, 0.25, 0.08], [0.00, 0.12, 0.35]]),
                torch.randn(128, 3) @ torch.tensor([[0.30, 0.00, 0.10], [0.12, 0.42, 0.04], [0.02, 0.08, 0.28]])
                + torch.tensor([3.0, 2.0, 4.0]),
            ],
            dim=0,
        ).contiguous()

        for covariance_type in ["spherical", "diag", "tied", "full"]:
            model = FlashGMM(
                d=3,
                k=2,
                niter=3,
                tol=0.0,
                use_triton=False,
                seed=7,
                init_params="random",
                chunk_size_data=64,
                chunk_size_centroids=1,
                chunk_size_data_cpu=96,
                covariance_type=covariance_type,
                device=torch.device("cuda"),
            )
            labels_cpu = model.fit_predict(x)
            pred_cpu = model.predict(x)
            scores_cpu = model.score_samples(x)
            probs_cpu = model.predict_proba(x[:128])

            x_gpu = x.to(model.device)
            pred_gpu = model.predict(x_gpu).cpu()
            scores_gpu = model.score_samples(x_gpu).cpu()
            probs_gpu = model.predict_proba(x[:128].to(model.device)).cpu()

            self.assertTrue(model.large_n_streaming_enabled_)
            self.assertEqual(labels_cpu.shape, (256,))
            self.assertEqual(pred_cpu.shape, (256,))
            self.assertEqual(scores_cpu.shape, (256,))
            self.assertEqual(probs_cpu.shape, (128, 2))
            self.assertTrue(torch.equal(pred_cpu, pred_gpu))
            self.assertTrue(torch.allclose(scores_cpu, scores_gpu, atol=1e-4, rtol=1e-4))
            self.assertTrue(torch.allclose(probs_cpu, probs_gpu, atol=1e-4, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton inference kernels")
    def test_diag_and_tied_triton_inference_match_torch(self):
        torch.manual_seed(8)
        x = torch.cat(
            [
                torch.randn(128, 16) * 0.4,
                torch.randn(128, 16) * 0.6 + 2.0,
            ],
            dim=0,
        ).contiguous()
        x_gpu = x.cuda()

        diag_model = FlashGMM(
            d=16,
            k=4,
            niter=3,
            tol=0.0,
            use_triton=True,
            seed=8,
            init_params="random",
            covariance_type="diag",
            device=torch.device("cuda"),
        ).fit(x_gpu)
        self.assertTrue(diag_model._use_triton_diag_inference(x_gpu.unsqueeze(0)))
        diag_labels = diag_model.predict(x_gpu)
        diag_probs = diag_model.predict_proba(x_gpu[:64])
        diag_scores = diag_model.score_samples(x_gpu[:64])
        diag_ref_labels = diagonal_assign_torch_native_chunked(
            x_gpu.unsqueeze(0),
            diag_model.means_b,
            diag_model.variances_b,
            diag_model.weights_b,
        ).squeeze(0)
        diag_ref_probs = diagonal_predict_proba_torch_native_chunked(
            x_gpu[:64].unsqueeze(0),
            diag_model.means_b,
            diag_model.variances_b,
            diag_model.weights_b,
        ).squeeze(0)
        diag_ref_scores = diagonal_score_samples_torch_native_chunked(
            x_gpu[:64].unsqueeze(0),
            diag_model.means_b,
            diag_model.variances_b,
            diag_model.weights_b,
        ).squeeze(0)
        self.assertTrue(torch.equal(diag_labels, diag_ref_labels))
        self.assertTrue(torch.allclose(diag_probs, diag_ref_probs, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(diag_scores, diag_ref_scores, atol=1e-4, rtol=1e-4))

        tied_model = FlashGMM(
            d=16,
            k=4,
            niter=3,
            tol=0.0,
            use_triton=True,
            seed=8,
            init_params="random",
            covariance_type="tied",
            device=torch.device("cuda"),
        ).fit(x_gpu)
        self.assertTrue(tied_model._use_triton_tied_inference(x_gpu.unsqueeze(0)))
        tied_probs = tied_model.predict_proba(x_gpu[:64])
        tied_scores = tied_model.score_samples(x_gpu[:64])
        tied_ref_probs = tied_predict_proba_torch_native_chunked(
            x_gpu[:64].unsqueeze(0),
            tied_model.means_b,
            tied_model.variances_b,
            tied_model.weights_b,
        ).squeeze(0)
        tied_ref_scores = tied_score_samples_torch_native_chunked(
            x_gpu[:64].unsqueeze(0),
            tied_model.means_b,
            tied_model.variances_b,
            tied_model.weights_b,
        ).squeeze(0)
        self.assertTrue(torch.allclose(tied_probs, tied_ref_probs, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(tied_scores, tied_ref_scores, atol=1e-4, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for large-CPU Triton inference")
    def test_large_cpu_triton_inference_helpers_match_torch(self):
        torch.manual_seed(9)

        def make_spd(batch: int, k: int, d: int) -> torch.Tensor:
            a = torch.randn(batch, k, d, d, device="cuda")
            eye = torch.eye(d, device="cuda").view(1, 1, d, d)
            return torch.matmul(a, a.transpose(-1, -2)) / float(d) + 0.5 * eye

        cases = [
            ("spherical", 16, 8),
            ("diag", 16, 8),
            ("tied", 16, 8),
            ("full", 4, 8),
        ]
        for covariance_type, d, k in cases:
            x = torch.randn(1, 384, d, dtype=torch.float32)
            means = torch.randn(1, k, d, device="cuda")
            weights = torch.rand(1, k, device="cuda")
            weights = weights / weights.sum(dim=-1, keepdim=True)
            if covariance_type == "spherical":
                variances = torch.rand(1, k, device="cuda") + 0.5
            elif covariance_type == "diag":
                variances = torch.rand(1, k, d, device="cuda") + 0.5
            elif covariance_type == "full":
                variances = make_spd(1, k, d)
            else:
                variances = make_spd(1, 1, d)[:, 0]

            kwargs = {
                "covariance_type": covariance_type,
                "device": torch.device("cuda"),
                "dtype": torch.float32,
                "chunk_size_N": 128,
                "chunk_size_K": 4,
            }
            labels_triton = large_n_predict_cpu(
                x,
                means,
                variances,
                weights,
                use_triton=True,
                **kwargs,
            )
            labels_torch = large_n_predict_cpu(
                x,
                means,
                variances,
                weights,
                use_triton=False,
                **kwargs,
            )
            scores_triton = large_n_score_samples_cpu(
                x,
                means,
                variances,
                weights,
                use_triton=True,
                **kwargs,
            )
            scores_torch = large_n_score_samples_cpu(
                x,
                means,
                variances,
                weights,
                use_triton=False,
                **kwargs,
            )
            probs_triton = large_n_predict_proba_cpu(
                x,
                means,
                variances,
                weights,
                use_triton=True,
                **kwargs,
            )
            probs_torch = large_n_predict_proba_cpu(
                x,
                means,
                variances,
                weights,
                use_triton=False,
                **kwargs,
            )

            self.assertTrue(torch.equal(labels_triton, labels_torch), covariance_type)
            self.assertTrue(torch.allclose(scores_triton, scores_torch, atol=1e-4, rtol=1e-4), covariance_type)
            self.assertTrue(torch.allclose(probs_triton, probs_torch, atol=1e-4, rtol=1e-4), covariance_type)
            self.assertTrue(
                torch.allclose(
                    probs_triton.sum(dim=-1),
                    torch.ones_like(scores_triton),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                covariance_type,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for fused update kernels")
    def test_fused_update_kernels_match_torch(self):
        if triton_fused_single_tile_update_spherical is None:
            self.skipTest("Triton fused update kernels unavailable")
        torch.manual_seed(10)
        bsz, n, d, k = 1, 160, 16, 8
        x = torch.randn(bsz, n, d, device="cuda")
        means = torch.randn(bsz, k, d, device="cuda")
        weights = torch.softmax(torch.randn(bsz, k, device="cuda"), dim=-1)
        log_weights = torch.log(weights)
        config = fused_single_tile_update_config(d, k)
        self.assertIsNotNone(config)

        variances = torch.rand(bsz, k, device="cuda") + 0.5
        x_sq = x.square().sum(dim=-1)
        means_sq = means.square().sum(dim=-1)
        logits = _compute_chunk_logits(x, x_sq, means, variances, log_weights)
        log_norm = torch.logsumexp(logits, dim=-1)
        resp = torch.exp(logits - log_norm.unsqueeze(-1))
        nk_ref = resp.sum(dim=1)
        sum_x_ref = torch.bmm(resp.transpose(1, 2), x)
        sum_x_sq_ref = (resp * x_sq.unsqueeze(-1)).sum(dim=1)
        nk, sum_x, sum_x_sq, ll = triton_fused_single_tile_update_spherical(
            x,
            means,
            variances,
            weights,
            x_sq=x_sq,
            means_sq=means_sq,
            log_weights=log_weights,
            **config,
        )
        self.assertTrue(torch.allclose(nk, nk_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(sum_x, sum_x_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(sum_x_sq, sum_x_sq_ref, atol=1e-3, rtol=1e-4))
        self.assertLess(abs(float(ll.item() - log_norm.sum().item())), 5e-3)

        diag_vars = torch.rand(bsz, k, d, device="cuda") + 0.5
        precision = diag_vars.reciprocal()
        weighted_means = means * precision
        mean_precision_mean = (means * weighted_means).sum(dim=-1)
        logdet = torch.log(diag_vars).sum(dim=-1)
        logits = _compute_diag_chunk_logits(
            x,
            means,
            diag_vars,
            log_weights,
            precision_chunk=precision,
            logdet_chunk=logdet,
            weighted_means_chunk=weighted_means,
            mean_precision_mean_chunk=mean_precision_mean,
        )
        log_norm = torch.logsumexp(logits, dim=-1)
        resp = torch.exp(logits - log_norm.unsqueeze(-1))
        nk_ref = resp.sum(dim=1)
        sum_x_ref = torch.bmm(resp.transpose(1, 2), x)
        sum_x_sq_ref = torch.bmm(resp.transpose(1, 2), x.square())
        nk, sum_x, sum_x_sq, ll = triton_fused_single_tile_update_diag(
            x,
            precision,
            weighted_means,
            mean_precision_mean,
            logdet,
            log_weights,
            **config,
        )
        self.assertTrue(torch.allclose(nk, nk_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(sum_x, sum_x_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(sum_x_sq, sum_x_sq_ref, atol=1e-3, rtol=1e-4))
        self.assertLess(abs(float(ll.item() - log_norm.sum().item())), 5e-3)

        a = torch.randn(bsz, d, d, device="cuda")
        tied_cov = torch.bmm(a, a.transpose(1, 2)) / float(d) + 0.5 * torch.eye(d, device="cuda").unsqueeze(0)
        tied_precision, tied_logdet = _precision_and_logdet(tied_cov)
        precision_means = torch.bmm(means, tied_precision.transpose(1, 2))
        mean_precision_mean = (means * precision_means).sum(dim=-1)
        logits = _compute_tied_chunk_logits(
            x,
            means,
            tied_cov,
            log_weights,
            precision=tied_precision,
            logdet=tied_logdet,
            precision_means_chunk=precision_means,
            mean_precision_mean_chunk=mean_precision_mean,
        )
        log_norm = torch.logsumexp(logits, dim=-1)
        resp = torch.exp(logits - log_norm.unsqueeze(-1))
        nk_ref = resp.sum(dim=1)
        sum_x_ref = torch.bmm(resp.transpose(1, 2), x)
        chol_precision = torch.linalg.cholesky(tied_precision)
        means_projected = torch.bmm(means, chol_precision)
        nk, sum_x, ll = triton_fused_single_tile_update_tied_native(
            x,
            chol_precision,
            means_projected,
            means_projected.square().sum(dim=-1),
            tied_logdet,
            log_weights,
            **config,
        )
        self.assertTrue(torch.allclose(nk, nk_ref, atol=1e-3, rtol=1e-4))
        self.assertTrue(torch.allclose(sum_x, sum_x_ref, atol=1e-3, rtol=1e-4))
        self.assertLess(abs(float(ll.item() - log_norm.sum().item())), 5e-3)


if __name__ == "__main__":
    unittest.main()
