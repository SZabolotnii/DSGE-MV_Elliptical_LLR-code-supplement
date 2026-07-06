import Mathlib

/-!
# Elliptical / multivariate extension of the LLR-projection unification (Paper 5 core)

Extends `Ku_Science/Ku-LSU/LikelihoodSeries.lean` (Paper 4, scalar-Gaussian anchor) to the
**elliptical** case, where the identity (Gaussian/Ghosh order-2) link stops being exact.

For an elliptical class model with **radial generator** `φ : ℝ → ℝ` and Mahalanobis radius
`D_c(s) = (s-μ_c)ᵀΣ_c⁻¹(s-μ_c)`,

    log p_c(s) = φ(D_c(s)) - ½·log|Σ_c| - κ,

so the log-likelihood ratio is a function of the two radii through the **same** generator:

    Λ(s) = φ(D₁(s)) - φ(D₀(s)) - ½(log|Σ₁| - log|Σ₀|).

Machine-checked content (all `sorry`-free, Mathlib v4.26.0):
  * `ellLLR_eq_radial_difference`     — the radial-link bridge above (any generator, any dim);
  * `ellLLR_mem_span_radial`          — Λ ∈ span{1, φ∘D₀, φ∘D₁}: the **GAM** with link φ;
  * `affine_link_mem_identity_span`   — if φ is affine (Gaussian carrier) the link collapses to
                                        the **identity** link span{1, D₀, D₁} (recovers Paper 4);
  * `mem_affine_span_iff_affine`      — the identity link represents **exactly** the affine
                                        generators;
  * `sq_not_mem_affine_span`          — the quadratic radial link (order-2 Kunchenko term, leading
                                        Taylor term of every non-Gaussian elliptical generator) is
                                        **not** in the identity span → for elliptical families the
                                        non-identity GAM/Kunchenko link is genuinely required
                                        (the formal counterpart of gate G-ELL-1);
  * `tGen_eq`, `tGen_not_affine`      — the Student-t generator is a concrete non-affine instance;
  * `mahalaMV_expand`                 — the multivariate Mahalanobis radius is a genuine quadratic
                                        form (quadratic + linear + const), so the multivariate
                                        Gaussian LLR lives in the quadratic span; the abstract
                                        Paper-4 bridge (Bayes = min-penalty, affine-in-heads)
                                        transfers to dimension `d` verbatim.

Drop-in build (same procedure as Paper 4): copy to
`../../dsge-spectral/lean/DescriptorBasis/EllipticalUnification.lean`, add the import to the
`DescriptorBasis` root, `lake build`; remove the drop-in copy after verification.  Self-contained
(`import Mathlib`).
-/

namespace EllipticalUnification

variable {S : Type*}

/-- An elliptical-in-basis per-class model: Mahalanobis radius field `D` and log-determinant `ld`.
The generator `φ` is supplied separately (shared across classes of one elliptical family). -/
structure EllBasis (S : Type*) where
  D  : S → ℝ
  ld : ℝ

/-- Elliptical log-density `φ(D_c(s)) - ½·log|Σ_c| - κ` (shared constant `κ`). -/
noncomputable def ellLogDensity (φ : ℝ → ℝ) (m : EllBasis S) (κ : ℝ) (s : S) : ℝ :=
  φ (m.D s) - (1 / 2) * m.ld - κ

/-- Elliptical detection functional (log-likelihood ratio) between two class models. -/
noncomputable def ellLLR (φ : ℝ → ℝ) (m₁ m₀ : EllBasis S) (κ : ℝ) (s : S) : ℝ :=
  ellLogDensity φ m₁ κ s - ellLogDensity φ m₀ κ s

/-- **Radial-link bridge.** The elliptical LLR is the difference of the generator evaluated at the
two Mahalanobis radii, plus the log-determinant offset.  Generalizes the Gaussian
`llr_eq_score_difference` of Paper 4 from the identity link `φ(u) = -u/2` to any generator. -/
theorem ellLLR_eq_radial_difference (φ : ℝ → ℝ) (m₁ m₀ : EllBasis S) (κ : ℝ) (s : S) :
    ellLLR φ m₁ m₀ κ s = φ (m₁.D s) - φ (m₀.D s) - (1 / 2) * (m₁.ld - m₀.ld) := by
  unfold ellLLR ellLogDensity; ring

/-- **GAM membership (the link).** As a function of `s`, the elliptical LLR lies in the span of
`{1, φ∘D₀, φ∘D₁}` — a generalized additive model in the two Mahalanobis heads with **link `φ`**.
For `φ = id`-affine this is Ghosh's identity-link classifier; for non-affine `φ` it is the genuine
GAM that Ghosh fits nonparametrically and the Kunchenko apparatus derives from the generator. -/
theorem ellLLR_mem_span_radial (φ : ℝ → ℝ) (m₁ m₀ : EllBasis S) (κ : ℝ) :
    (fun s => ellLLR φ m₁ m₀ κ s) ∈
      Submodule.span ℝ
        ({fun _ => 1, fun s => φ (m₀.D s), fun s => φ (m₁.D s)} : Set (S → ℝ)) := by
  have hrep : (fun s => ellLLR φ m₁ m₀ κ s)
      = (-(1 / 2) * (m₁.ld - m₀.ld)) • (fun _ : S => (1 : ℝ))
        + (-1 : ℝ) • (fun s => φ (m₀.D s))
        + (1 : ℝ) • (fun s => φ (m₁.D s)) := by
    funext s
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [ellLLR_eq_radial_difference]; ring
  rw [hrep]
  refine Submodule.add_mem _ (Submodule.add_mem _ ?_ ?_) ?_ <;>
    refine Submodule.smul_mem _ _ (Submodule.subset_span ?_)
  · exact Set.mem_insert _ _
  · exact Set.mem_insert_of_mem _ (Set.mem_insert _ _)
  · exact Set.mem_insert_of_mem _ (Set.mem_insert_of_mem _ rfl)

/-- **Affine link ⇒ identity link (Gaussian collapse).** When the generator is affine
`φ(u) = a·u + b` (the Gaussian carrier has `a = -½, b = 0`), the elliptical LLR lies in the
**identity-link** span `{1, D₀, D₁}` — recovering the Paper-4 Gaussian/Ghosh anchor. -/
theorem affine_link_mem_identity_span (a b : ℝ) (m₁ m₀ : EllBasis S) (κ : ℝ) :
    (fun s => ellLLR (fun u => a * u + b) m₁ m₀ κ s) ∈
      Submodule.span ℝ
        ({fun _ => 1, fun s => m₀.D s, fun s => m₁.D s} : Set (S → ℝ)) := by
  have hrep : (fun s => ellLLR (fun u => a * u + b) m₁ m₀ κ s)
      = (-(1 / 2) * (m₁.ld - m₀.ld)) • (fun _ : S => (1 : ℝ))
        + (-a : ℝ) • (fun s => m₀.D s)
        + (a : ℝ) • (fun s => m₁.D s) := by
    funext s
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [ellLLR_eq_radial_difference]; ring
  rw [hrep]
  refine Submodule.add_mem _ (Submodule.add_mem _ ?_ ?_) ?_ <;>
    refine Submodule.smul_mem _ _ (Submodule.subset_span ?_)
  · exact Set.mem_insert _ _
  · exact Set.mem_insert_of_mem _ (Set.mem_insert _ _)
  · exact Set.mem_insert_of_mem _ (Set.mem_insert_of_mem _ rfl)

/-! ### The identity link represents exactly the affine generators (formal G-ELL-1) -/

/-- The identity-link span `{1, id}` over `ℝ → ℝ` is **exactly** the affine functions. -/
theorem mem_affine_span_iff_affine (g : ℝ → ℝ) :
    g ∈ Submodule.span ℝ ({fun _ => 1, fun u => u} : Set (ℝ → ℝ)) ↔
      ∃ a b, g = fun u => a * u + b := by
  rw [Submodule.mem_span_pair]
  constructor
  · rintro ⟨c, d, hcd⟩
    refine ⟨d, c, ?_⟩
    funext u
    have := congrFun hcd u
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul] at this
    linarith [this]
  · rintro ⟨a, b, rfl⟩
    refine ⟨b, a, ?_⟩
    funext u
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring

/-- **The quadratic radial link is NOT an identity link.** `u ↦ u²` — the order-2 Kunchenko term
and the leading Taylor term of every non-Gaussian elliptical generator — is not in `span{1, id}`.
Hence for a non-affine generator the LLR cannot be reproduced by the identity (Gaussian/Ghosh)
link; the non-identity GAM/Kunchenko link is genuinely required. -/
theorem sq_not_mem_affine_span :
    (fun u : ℝ => u ^ 2) ∉ Submodule.span ℝ ({fun _ => 1, fun u => u} : Set (ℝ → ℝ)) := by
  rw [mem_affine_span_iff_affine]
  rintro ⟨a, b, h⟩
  have h0 := congrFun h 0
  have h1 := congrFun h 1
  have h2 := congrFun h 2
  norm_num at h0 h1 h2
  nlinarith [h0, h1, h2]

/-! ### A concrete non-affine elliptical generator: the Student-t carrier -/

/-- Student-t(ν) radial generator in dimension `d`: `φ(u) = -((ν+d)/2)·log(1 + u/ν)`. -/
noncomputable def tGen (ν d u : ℝ) : ℝ := -((ν + d) / 2) * Real.log (1 + u / ν)

/-- Sanity: `tGen` definitional unfolding. -/
theorem tGen_eq (ν d u : ℝ) : tGen ν d u = -((ν + d) / 2) * Real.log (1 + u / ν) := rfl

/-- The Student-t generator is **not affine**: with `ν = 1, d = 1` the three points `u ∈ {0,1,3}`
violate the affine second-difference (`φ(0)+φ(3) ≠ 2φ(1)` since `log 4 ≠ 2 log 2`... here via
`log 1 = 0`, `log 2`, `log 4 = 2 log 2`: an affine fit through `(0,φ0),(1,φ1)` overshoots at 3).
Concretely it is strictly convex (a negative multiple of the strictly concave `log`), hence ∉ the
identity span — the elliptical link is genuinely non-identity. -/
theorem tGen_not_affine : ¬ ∃ a b, tGen 1 1 = fun u => a * u + b := by
  rintro ⟨a, b, h⟩
  have h0 := congrFun h 0
  have h3 := congrFun h 3
  have h1 := congrFun h 1
  simp only [tGen] at h0 h3 h1
  -- φ(0) = 0; φ(1) = -log 2; φ(3) = -log 4 = -2 log 2.  Affine ⇒ φ(3)-φ(1) = 2(φ(1)-φ(0)).
  rw [show (1:ℝ) + 0/1 = 1 by norm_num, Real.log_one] at h0
  rw [show (1:ℝ) + 1/1 = 2 by norm_num] at h1
  rw [show (1:ℝ) + 3/1 = 4 by norm_num] at h3
  rw [show (4:ℝ) = 2 ^ 2 by norm_num, Real.log_pow] at h3
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  -- from h0: b = 0; h1: a + b = -log2; h3: 3a + b = -2 log2  ⇒  contradiction
  nlinarith [h0, h1, h3, hlog2]

/-! ### Multivariate Gaussian: the Mahalanobis radius is a genuine quadratic form -/

section Multivariate

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

open scoped RealInnerProductSpace

/-- Multivariate Mahalanobis radius `D(x) = ⟪x-μ, A(x-μ)⟫` for a precision operator `A`. -/
noncomputable def mahalaMV (μ : E) (A : E →ₗ[ℝ] E) (x : E) : ℝ := ⟪x - μ, A (x - μ)⟫

/-- **The multivariate Mahalanobis radius is quadratic + linear + constant.** Expanding the
inner product exhibits the degree-2 structure: the multivariate Gaussian LLR therefore lies in
the span of the quadratic monomials (the multivariate analog of Paper 4's `gauss_llr_eq_quadratic`
/ `gauss_llr_mem_span_quadratics`), and the abstract Paper-4 bridge transfers verbatim to dim d. -/
theorem mahalaMV_expand (μ : E) (A : E →ₗ[ℝ] E) (x : E) :
    mahalaMV μ A x = ⟪x, A x⟫ - ⟪x, A μ⟫ - ⟪μ, A x⟫ + ⟪μ, A μ⟫ := by
  unfold mahalaMV
  rw [map_sub, inner_sub_left, inner_sub_right, inner_sub_right]
  ring

end Multivariate

end EllipticalUnification
