# OpenReview Thread for Submission 1678

## Official Review of Submission1678 by Reviewer 3Msq

**Official Review by Reviewer 3Msq**  
*10 Nov 2025, 21:36 (modified: 12 Dec 2025, 23:27)*    
Status: Submitted  
Reviewer: 3Msq

### Summary And Contributions
The authors propose a framework for multi-marginal flow matching. The two main contributions are using a piecewise quadratic conditional path for the flow matching training, as well as learning the diffusion coefficient to predict the uncertainty based on the deviation of the prediction at the next time interval based on the current learned velocity. The authors argue that the former contributes to smoother target paths instead of the naive piecewise constant paths, as the novel quadratic term contributes information about future velocities. However in contrast to other spline-based methods there is no computational overhead and the authors claim the method is more stable in higher dimensions. The latter in principle should help quantify uncertainty. For unpaired trajectory data the authors use a previously reported scheme based on optimal transport. The authors benchmark their method against other SOTA methods on synthetic and real world data sets.

**Paper Keywords:** time series, flow matching, trajectory modeling  
**Expertise Keywords:** time series, flow matching, disease progression modeling

### Soundness
**Assessment:** Correct / minor errors (e.g., typo or errors that do not affect the main results)

**Justification:**  
The authors provide proofs of their claims. The main claims are the suggested velocity and the derivation thereof as well as a proof that the additional loss term to learn uncertainty at the same time does not change the learned trajectory (the stationary points are the same).

### Significance
**Assessment:** Significant contributions (e.g., strong theoretical insights or well-supported empirical improvements with appropriate baselines/statistics).

**Justification:**  
The two main contributions of the paper are practical contributions with theoretical justifications to make multi-marginal flow matching models easier to learn and produce better results. The piecewise quadratic path is a nice and easy way to make the target path more continuous using the average velocity, while not having to resolve to fitting splines over the full interval or calculating the instantaneous velocity for local Hermite splines. Uncertainty quantification is important for many applications and while learning the diffusion term as a difference between drift and data is not novel in the literature of heteroscedastic uncertainty estimation, to the best of my knowledge it's the first application in the field of flow matching, together with a theoretical justification that it does not change the original trajectory. It would have been nice however to quantify the quality of the uncertainty quantification.

### Novelty
**Assessment:** New results

**Justification:**  
As written above, the paper contributes new results (albeit not ground-breaking) to the multi-marginal flow matching literature, mainly based on practical contributions to empirically improve training and applications.

### Non-conventional Contributions
N/A

### Clarity
The paper is clearly written and clearly highlights and justifies its contributions. The examples and data sets are relevant.

### Relation To Prior Work
All related works are clearly discussed.

### Additional Comments
- One of the main claims of the paper is being able to do uncertainty quantification, it would have therefore been good to have experiments quantifying the calibration of the estimated uncertainty.
- Is it necessary that all realizations of the time series (e.g. patients in the disease progression) are measured at the same time points? This could be a bit more explicit (or explained how this can be incorporated).

### Reproducibility
Sufficient amount of details available for reproducing the main results.

### Rating and Confidence
- **Rating:** 6: Accept (Technically solid paper, with high impact on at least one sub-area. The contribution is convincing and the evaluation is adequate, though there may be some limitations or open questions.)
- **Confidence:** 3: Fairly confident (It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.)

---

## Rebuttal by Authors (to Reviewer 3Msq)

**Rebuttal by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*28 Nov 2025, 16:32 (modified: 01 Dec 2025, 14:11)*    
Status: Submitted

### Rebuttal

1. **Quantifying uncertainty quality and uncertainty calibration experiments.**

We appreciate the reviewer's suggestion and agree that uncertainty calibration is an important question in settings where the uncertainty term is intended to represent a predictive variance. However, in our method, the learned uncertainty plays a fundamentally different role. This makes classical calibration analyses not directly applicable.

In our SDE formulation, the uncertainty term is learned jointly with the prediction and is used to update the latent-space state toward the final solution. It is therefore part of the prediction mechanism, not a separate estimate of uncertainty in the classical sense. Instead of directly representing the model's uncertainty, the uncertainty term guides the SDE trajectory towards a better reconstruction. Because of this, we do not interpret this term as a standard deviation or confidence interval; in our opinion, this makes the calibration plots not meaningful in our context. To summarize: the learned uncertainty term is not a predictive variance but a directional corrective term.

We also want to note that the SDE is modeled in latent space, meaning that the uncertainty is a high-dimensional (4096 in the case of ADNI) latent vector. This latent space is not physically interpretable. The meaning of moving one unit in latent space in any given direction would not be interpretable in the final image domain since the latent space is not guaranteed to be disentangled.

Regarding the uncertainty term's quality, it consistently improves the final prediction compared to the models that do not include this term. This already suggests that the term is aligned with the model's prediction error. If not, it would not systematically reduce the error. This means that *the effect of the learned term already appears in the improved predictive performance*.

To avoid this confusion, we add to the Appendix G.2 a dedicated section that clearly explains that the uncertainty term is a learned directional correction in latent space, rather than a probabilistic uncertainty estimate in the classical sense. We hope that the above provides sufficient context for the uncertainty term. In the case we misunderstood the question, we would be very interested to hear ideas for evaluating the calibration of the uncertainty term.

2. **On alignment of time points across trajectories.**

We appreciate this question, and we agree that the alignment of time point measurements of the time series should be explained explicitly.

We do not require that all realizations are measured at the same time points. We only assume that each observation is sampled from a continuous process over time, which we demonstrated on the synthetic Gaussian, starmen, and other datasets. Note that for the experimental results on the ADNI dataset, we bin the measurements to 6-month intervals to facilitate better (and more clinically relevant) analysis, but the actual measurements are not imaged at those exact times. We now explicitly mention in Section 2.0 (Background) that we do not require all realizations to be measured at the same time points.

---

## Official Comment by Reviewer 3Msq

**Official Comment by Reviewer 3Msq**  
*08 Dec 2025, 02:00*    
Status: Submitted

**Comment:**  
Thank you for the clarification and pointing out my misunderstanding. The authors addressed my concerns.

---

## Official Comment by Authors (to Reviewer 3Msq)

**Official Comment by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*08 Dec 2025, 16:53*    
Status: Submitted

**Comment:**  
We would like to thank the reviewer for their insightful review and discussion. We are glad to hear that we were able to address all of your concerns. We would kindly ask if you would consider upgrading your original score.

---

## Author Review Rating of Submission1678 by Mohammad Mohaiminul Islam (for Reviewer 3Msq)

**Author Review Rating by** Mohammad Mohaiminul Islam  
*28 Dec 2025, 15:23*  eets expectations

---

## Official Review of Submission1678 by Reviewer Y8jE

**Official Review by Reviewer Y8jE**  
*09 Nov 2025, 13:02 (modified: 12 Dec 2025, 23:27)*    
Status: Submitted  
Reviewer: Y8jE

### Summary And Contributions
A novel generative method for trajectory data is proposed for when the data is high-dimensional and sparsely sampled. The method considers trajectories as a consistent whole, representing the problem as learning continuous stochastic dynamics over iterated pairwise transitions, sometimes favoured by existing work, and further aims for subject-specific dynamics over population-wide synthesis. These targets are achieved by adopting a differential equation-based framework, interpolating between points in a trajectory with a smooth velocity transition, and a theoretically derived training objective. The key contributions of the work are 1) the proposed method IMMFM, which offers full image stochastic dynamics as well as subject-specific dynamics, while existing work typically only supports one of these properties, 2) the theoretical derivation of a suitable optimisation target, and 3) empirical experimental validation on both synthetic and real-world medical imaging data.

**Paper Keywords:** trajectory data, flow matching, generative models  
**Expertise Keywords:** trajectory data, spatial interpolation, empirical research methods

### Soundness
**Assessment:** Correct / minor errors (e.g., typo or errors that do not affect the main results)

**Justification:**  
I could not find any major errors. Although I have some comments on the empirical methods (see "significance" and "clarity"), I would not consider these large enough to form a problem for technical soundness.

### Significance
**Assessment:** Somewhat significant (e.g., theoretical contribution of limited novelty or empirical gains missing baselines/statistical rigor).

**Justification:**  
The authors address a clearly described gap in the state-of-the-art. The method and training objective are theoretically supported, and the focus on differential equations (though this is not, on its own, novel) lends credibility to the resulting generative models. On the other hand, the motivation for the selection of the three baseline methods is quite brief, while 5 out of the 8 result columns for trajectory forecasting in Table 1 consist of different variants of the proposed method. It is, therefore, not clear to me why different types of forecasting methods (e.g., traditional, non-generative forecasting methods) are not relevant to this comparison, while the inclusion of so many variants of the proposed methods (compared to baseline methods, which only get one shot each) can give an impression of an unfair comparison. Given the strong performance of SU-IMMFM, and its description as the most complete version including all the described contributions (if I understand correctly), I would consider it more appropriate to only compare to SU-IMMFM, possibly against non-generative forecasting methods, unless the authors motivate why their inclusion would be inappropriate, and compare IMMFM versions among themselves in a separate table. I currently also cannot find whether results marked bold are best as determined by a suitable statistical test, or only the best value without significance; I would recommend adding this information to the table header (and, if significance was not yet tested for, including these tests).

### Novelty
**Assessment:** New results

**Justification:**  
The work advances the state-of-the-art by addressing a specific, clearly described gap in the discussed related work. The contributions seem greater than incremental, as they introduce a combination of properties that were not previously available, but not ground-breaking, because it is not the first time generative models have been applied to trajectory data. Therefore, I consider this work "new results".

### Non-conventional Contributions
N/A

### Clarity
(1) The contributions, notation and results were clearly stated. (2) The claims made in the title, abstract and introduction were supported with results, although some of the detailed results are contained in the Appendix rather than the main text. (3) The meaning of theoretical assumptions was clear, but not necessarily explained in detail (though I did not miss this and don't consider further elaboration necessary). (4) The proposed method was motivated well, including with examples of where existing methods fall short.

Additional comments on clarity:
- The order of some figures is quite confusing. For example, in Section 5.1, the authors refer to Figure 3 as a result, which was shown 2 pages before to this reference (without prior reference in the text), and followed by Figure 4 and Table 1 in between. Since Figure 3 is discussed after Table 1, I would not normally expect to find it a page before Table 1.
- The text also frequently refers to figures and tables contained in the appendix. While this can be acceptable in places (e.g., "see also temporal generalizability in Appendix F.2"), it is less ideal in others (e.g., "We examine the contributions [...] compared to 2D.", Section 5.2), where it can feel like results supporting core claims by the authors are not included in the main text.

### Relation To Prior Work
All related works are clearly discussed.

### Additional Comments
I am aware that many of the comments I raised above were likely a result of the page limit, so perhaps some of them cannot be helped. I appreciate that the appendices contain many additional details, even if some of those might have benefited from being included in the main text.

### Reproducibility
Sufficient amount of details available for reproducing the main results.

### Rating and Confidence
- **Rating:** 6: Accept (Technically solid paper, with high impact on at least one sub-area. The contribution is convincing and the evaluation is adequate, though there may be some limitations or open questions.)
- **Confidence:** 3: Fairly confident (It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.)

---

## Rebuttal by Authors (to Reviewer Y8jE)

**Rebuttal by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*28 Nov 2025, 16:36 (modified: 01 Dec 2025, 14:11)*    
Status: Submitted

### Rebuttal

1. **On baselines and IMMFM variants in Table 1.**

We appreciate the reviewer's suggestion. First, we would like to clarify that only 3 out of the 8 results are variants of our proposed method (indicated in bold). This means we compare against 5 baselines (not only 3). To avoid further confusion, we now explicitly mention that the bolded names are our proposed models in the Table 1 caption.

Our choice to present all variants and baselines in a single table was intentional. The different IMMFM variants (e.g., quadratic path, quadratic path + SDE) represent independent modeling choices, rather than ablations leading to a single final model. Depending on the application and computational budget, different IMMFM variants may be preferable. For this reason, we believe it to be necessary to directly compare all variants to the baselines.

To put this in perspective: the SDE variants have an additional head to predict the \(\sigma\) term, as well as additional score matching, adding to the model (computational) complexity. This extra complexity is not always warranted. For example, in cases where the underlying processes are less stochastic and more smooth, such as in the starmen dataset (where it describes a 3-dimensional manifold representing the two arm movements and combined leg movement), it would be better to use the simpler ODE variant. We indeed see that this is the case in Table 1, where the ODE version of our model performs better on starmen compared to the SDE version.

We revise the text in Section 4 to clarify that the IMMFM-SU is not the only proposed model. Instead, all IMMFM configurations presented in the paper correspond to valid modeling choices.

2. **On including non-generative forecasting baselines.**

We selected our baselines from the family of state-of-the-art generative forecasting models because these models are conceptually closest to ours and are directly comparable. We agree that non-generative forecasting models form an important class of methods. One of our baselines, ImageFlowNet (Chen et al., 2025), evaluates between generative and non-generative approaches, and reports that their proposed model outperforms the non-generative models. Our model outperforms ImageFlowNet in our experiments, which indicates (by transitivity) that our approach would also surpass the non-generative methods tested by Chen et al. (2025).

We now reference to the non-generative forecasting models in Section 4 (Baselines).

3. **On figure ordering and placement of key results.**

We thank the reviewer for pointing these out. We have now reordered all figures and tables to appear in the correct order. Regarding the frequent references to the appendix, we placed all essential results that directly support the core claims in the main paper body. We do provide extended analyses, ablations, and qualitative examples in the appendix. This allowed us to maintain clarity in the main text without omitting important results. However, if there are any specific results from the appendix that you would like us to move for better clarity to the main paper body, we are happy to revise their placement accordingly.

4. **On the significance of bolded results.**

Thank you for raising this point. We realize that our current presentation may have been unclear.

All reported results are averaged over three runs and three random seeds (i.e., nine runs in total) for models involving SDE components. The standard deviations reported in the tables are inter-subject standard deviations, which are comparable across models. Given this setup, we think that performing an additional statistical test would not provide meaningful extra insight: the mean differences between models are already mostly larger than the relatively stable standard deviations across all methods (averaged over all time points, runs, and seeds) on multiple datasets and metrics. For this reason, boldface entries do not indicate statistical significance. Instead, bold simply marks the numerically best mean value in each column.

We agree that rounding of the results may have made this ambiguous. In the revised version, we now explicitly clarify what bold formatting represents in the Table 1 caption and have resolved inconsistencies in bolded entries in Table 1.

---

## Official Comment by Authors (follow-up to Reviewer Y8jE)

**Official Comment by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*08 Dec 2025, 16:54*    
Status: Submitted

**Comment:**  
We would like to thank the reviewer for reviewing our work and their valuable feedback. If you are happy with our answers or you would like further clarifications, please let us know.

---

## Official Comment by Reviewer Y8jE (reply)

**Official Comment by Reviewer Y8jE**  
*08 Dec 2025, 17:33*    
Status: Submitted

**Comment:**  
I am happy with the answers. Thank you very much for the clarification.

---

## Official Comment by Authors (final follow-up to Reviewer Y8jE)

**Official Comment by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*09 Dec 2025, 11:52*    
Status: Submitted

**Comment:**  
Thank you for letting us know that we clarified your concerns. If this affects your original evaluation, we would be thankful if you could reflect this in your original review score.

---

## Author Review Rating of Submission1678 by Mohammad Mohaiminul Islam (for Reviewer Y8jE)

**Author Review Rating by** Mohammad Mohaiminul Islam  
*28 Dec 2025, 15:23*  eets expectations

---

## Official Review of Submission1678 by Reviewer eJLn

**Official Review by Reviewer eJLn**  
*06 Nov 2025, 18:57 (modified: 12 Dec 2025, 23:27)*    
Status: Submitted  
Reviewer: eJLn

### Summary And Contributions
The authors propose a generative model within the Flow Matching and simulation-free diffusion framework for forecasting continuous stochastic dynamical systems from sequential data. Their approach is designed for sparse, irregular, and noisy samples, and is compared against well-chosen baselines on both low and high-dimensional datasets. The main contribution extends prior works on simulation-free SDE modelling, such as Trajectory Flow Matching (Zhang et al. 2024), by introducing a quadratic piecewise interpolation scheme between consecutive observations. This enables smoother modelling of curved dynamics where linear interpolation fails. Beyond empirical results on synthetic and real datasets, the paper provides theoretical justification for its joint optimisation of drift and diffusion terms, and incorporates an Optimal Transport-based alignment method to handle sparse and irregular trajectories, and find the optimal transport coupling between observations.

**Paper Keywords:** generative model, clinical data, dynamic systems  
**Expertise Keywords:** dynamics system, generative model, bayesian optimisation

### Soundness
**Assessment:** Correct / minor errors (e.g., typo or errors that do not affect the main results)

**Justification:**  
I did not find major conceptual or empirical errors. The proofs, derivations, and arguments presented in the paper appear correct at the level provided. The experiments and evaluations are also well conducted.

### Significance
**Assessment:** Significant contributions (e.g., strong theoretical insights or well-supported empirical improvements with appropriate baselines/statistics).

**Justification:**  
Although the method is an incremental extension of existing work as reported in the "Novelty" section, it introduces important improvements in handling irregular and sparse data through optimal transport alignment and a novel interpolation scheme for smoother dynamics assumptions. The method is likely to be of interest to researchers working on generative models for time series and trajectory data. The method is well described and justified both theoretically and empirically on relevant datasets.

### Novelty
**Assessment:** Incremental compared to existing results

**Justification:**  
The method comes as an extension of the Trajectory Flow Matching (Zhang et al, 2025) existing method and other concurrent recent works. However, the method is well described and shows empirical improvements on relevant datasets for clinical experimentation and shows the effectiveness also on high-dimensional data. The main contributions lie in the quadratic interpolation scheme and the optimal transport alignment for irregular and sparse data. The quadratic interpolation has been justified for learning smoother dynamics, and the proposed loss objective function has been theoretically justified in the paper by extending Theorem 3.2 of Zhang et al. Instead, the sparsity and irregularity handling is based on existing OT literature. For this reason, I assess this work as an incremental contribution to the existing literature.

### Non-conventional Contributions
No

### Clarity
The paper is very well written, easy to follow, precise, and the contributions are clearly stated. The methodology is well motivated, and the theoretical assumptions are explained. Moreover, they also show some concept of proof when learning smooth dynamics with sparse data, where linear interpolation fails. The experimental results support the claims made in the introduction and abstract.

However, I would suggest adding some text in the paper addressing limitations of the method in the conclusion, regarding the computational cost of the OT alignment for large datasets, which is applied as a pre-step before training the proposed method for learning the dynamics, and the possible limitations of the quadratic interpolation scheme when the dynamics are very complex or chaotic.

### Relation To Prior Work
All related works are clearly discussed.

### Additional Comments
The related work section is comprehensive and clearly discussed. However, I would also suggest comparing with another work on a generative model for time series data, such as *SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations* (Bartosh 2025), where they learned the dynamics system in a simulation-free manner but in the latent space. Potentially a baseline to the method, since your method also considers learning the latent dynamics system, as they are focused on.

### Reproducibility
Sufficient amount of details available for reproducing the main results.

### Rating and Confidence
- **Rating:** 6: Accept (Technically solid paper, with high impact on at least one sub-area. The contribution is convincing and the evaluation is adequate, though there may be some limitations or open questions.)
- **Confidence:** 4: Confident, but not absolutely certain. (It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.)

---

## Rebuttal by Authors (to Reviewer eJLn)

**Rebuttal by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*28 Nov 2025, 16:47 (modified: 01 Dec 2025, 14:11)*    
Status: Submitted

### Rebuttal

1. **On limitations: OT alignment cost.**

We appreciate the reviewer's question regarding the practicality and computational cost of the OT alignment step.

For the imaging datasets, we register all timepoints prior to model training. Using the ANTs (https://github.com/ANTsX/ANTs) registration code, the full preprocessing pipeline for the ADNI dataset takes approximately 6 hours at a resolution of \(128^3\) and 1 hour for \(64^3\). For GBM, preprocessing takes roughly 9 hours for \(128^3\) and 2.5 hours for \(64^3\). These times are measured on an Intel Xeon Gold 5118 (released in 2017) CPU running at 2.30Ghz.

Registering a single volume is very fast. Roughly 6.85 seconds for \(128^3\) and 1.12 seconds for \(64^3\). These costs make the approach practical for real-time inference. We will add these timings to the revised manuscript.

We do want to emphasize that MRI images follow standardized protocols and are already reasonably aligned. As a result, the registration requires only a few iterations to converge. In imaging domains without such protocols, the misalignment could be larger. Consequently, registration would require more iterations and increase the cost.

Also note that with GPU-based registration tools and running the preprocessing pipeline in parallel these timings could be significantly improved for extremely large scale datasets. We used ANTs to keep the preprocessing as close as possible to the baselines.

We now added the cost of the OT alignment to the Appendix Section E.2 (Dataset Preprocessing and Augmentation).

2. **On limitations: quadratic spline interpolation.**

We thank the reviewer for this question, as limitations of the quadratic path construction are indeed important to consider. Below, we discuss potential limitations of our formulation.

We identify the primary limitation of the quadratic interpolation scheme in chaotic regimes to be its low-pass filtering effect. Because the conditional path is constructed as a piecewise-quadratic segment (resulting in velocity evolution as derived in Appendix A.1, Eq. 46), it effectively models the local dynamics as a smooth trajectory with stable curvature. Consequently, the method may filter out high-frequency oscillations or jerky changes (non-zero third derivative) that occur precisely between sparse observation points.

However, we argue that this behavior is a helpful feature for effective training rather than a strict drawback. In the presence of utterly chaotic or extremely stochastic dynamics, learning a continuous probability flow from sparse data becomes ill-posed. Without the implicit regularization provided by a smooth target path, the resulting vector field would likely violate the Lipschitz continuity required for stable flow matching, leading to overfitting or non-converging solutions (analogous to the Runge phenomenon in splines).

We also importantly note here, this does not necessarily imply that the complex variations characteristic of chaotic dynamics are completely lost. Instead, we hypothesize that our framework can captures some of these dynamics through learned diffusion \(\sigma\) in the SU-IMMFM case. Where the quadratic drift maintains a stable mean trend to ensure tractability, the missed chaotic volatility is absorbed by the learned diffusion component. As the true chaotic path deviates from the smooth quadratic target, the predictive error naturally increases.

We added an additional section in Appendix H that incorporates the above discussion on the possible limitations of the quadratic path construction.

3. **On adding SDE Matching as a baseline.**

We thank the reviewer for pointing out the recent SDE Matching (Bartosh et al., 2025) work, which is indeed highly relevant as it is conceptually close to our approach.

Because this work appeared during the development of our paper, it was not included in the original manuscript. We now include this method in the literature discussion in Section 1 (Introduction).

We also want to include SDE Matching as a baseline method in our paper. However, because our computing cluster is fully down for the entire rebuttal period due to emergency maintenance, we cannot guarantee that all results will be ready in time before the 8th December. We are exploring every available option to obtain these results and will add them to the discussion and updated manuscript as soon as they are available.

---

## Official Comment by Reviewer eJLn (follow-up)

**Official Comment by Reviewer eJLn**  
*04 Dec 2025, 11:03*    
Status: Submitted

**Comment:**  
I thank the reviewer for providing more transparent and additional information about the OT compute cost.

1-2.) If I am not mistaken, in the end the OT alignment (MMOT) mainly addresses the pairing of the cloud points between pair states of trajectories, correct? But does it play any role or help for irregular time steps of the trajectory? In your forecasting generative model, is time also learned as it is done in TFM, thus time is learned by regressing the next timestep? If not, how do you address the irregular timestamps by the quadratic interpolant? If yes, can you please give me an explanation, why it should work and what are the limitations on this side?

Indeed, having the SDE matching as an extra baseline would be a very valuable incremental. But I do acknowledge the time constraint and technical issue of the cluster.

---

## Official Comment by Authors (reply to Reviewer eJLn)

**Official Comment by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*05 Dec 2025, 15:15 (modified: 08 Dec 2025, 16:55)*    
Status: Submitted

**Comment:**  
Thank you for giving us the opportunity to offer more clarity on the OT alignment and how we address irregular timesteps with our approach, and how it contrasts with TFM.

### On OT alignment for irregular timesteps

It is indeed correct that the OT alignment is used to pair (spatial correspondence) between observations. For example, for the Gaussian datasets it pairs the point clouds (samples) from the marginal distributions, and for the imaging datasets it's the diffeomorphic registration (see Eq. 20). The alignment is independent of the time between the two consecutive states, so it is does not play a role or help with irregular time steps.

### On addressing irregular timesteps of our approach vs. TFM

TFM includes an auxiliary head to predict the "time to next observation" (\(\Delta t\)), whereas our approach does not. Below, we first detail TFM's approach to address irregular time intervals, so that we can contrast it with our own.

TFM models transitions between discrete and irregular observations where the time interval (\(\Delta t\)) itself is a random variable. The velocity magnitude that is used to transition from one state to the next fully depends on the time between these states (\(\Delta t\)). This means that this vector field is undefined at inference if the time interval is not known. Furthermore, TFM relies on target-prediction reparameterization for training, defining the target state as \(x_{t+\Delta t} = x_t + \Delta t \cdot v_t\). While the ground truth interval is available during training, relying on it creates a dependency on information that is unavailable at test time. Consequently, TFM is structurally required to use an auxiliary head to predict the time to the next observation (\(\Delta t\)).

Our approach instead decouples the dynamics from the sampling schedule. We address irregular sampling directly through the construction of the conditional probability path (the training target), defined analytically by the time-derivative of the interpolant: \(v(x, t) = \partial_t \gamma(t)\).

The quadratic interpolant we use (see Eq. 9 and 10) explicitly incorporates the interval length \(\Delta t\) into the calculation of the segment velocities (\(v_{i}\)) and the blending coefficient (\(\alpha_i(t)\)). In essence, by smoothing the transition between the velocity of the incoming segment \(v_{i-1}\) and the outgoing segment \(v_{i}\), we construct a target vector field that is continuous in time and is adjusted for the instantaneous local rate of change, regardless of how irregular the sampling intervals are. As a result, our model does not need to predict the time interval.

### On limitations of the quadratic interpolant

Regarding the specific limitations of our interpolation scheme, we refer to the discussion in our previous response (Point 2) where we address the low-pass filtering effect. In short, the smooth curvature dampens high-frequency changes in the paths, and because there is velocity blending that mixes velocities between two consecutive segments, it introduces a tiny error to the instantaneous local rate of change. However, we argue that this actually ensures the Lipschitz continuity that is necessary for stable training on the irregular time intervals. We do want to reiterate that we view the low-pass effect as a necessary trade-off to avoid ill-posed vector fields in sparse data regimes.

We hope that with the above, we have provided clarity on the role of OT alignment and how we address irregular time intervals compared to TFM.

---

## Official Comment by Authors (additional clarification to Reviewer eJLn)

**Official Comment by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*08 Dec 2025, 16:55*    
Status: Submitted

**Comment:**  
First of all, we would like to thank the reviewer for the insightful discussion on the implications of the OT alignment and the quadratic interpolation scheme, especially since it allowed us to provide a much more detailed and hopefully clear understanding of our proposed framework.

We also wanted to add a bit more detail to our previous answer on the limitations of the quadratic interpolant. In particular, we added the following lines to the "On limitations of the quadratic interpolant." paragraph in our previous comment in italics: "In short, the smooth curvature dampens high-frequency changes in the paths, and because there is velocity blending that mixes velocities between two consecutive segments, it introduces a tiny error to the instantaneous local rate of change."

In case you are satisfied with our answers or would like us to provide further details, please let us know.

---

## Author Review Rating of Submission1678 by Mohammad Mohaiminul Islam (for Reviewer eJLn)

**Author Review Rating by** Mohammad Mohaiminul Islam  
*28 Dec 2025, 15:23*  eets expectations

---

## Official Review of Submission1678 by Reviewer uDJv

**Official Review by Reviewer uDJv**  
*30 Oct 2025, 11:46 (modified: 12 Dec 2025, 23:27)*    
Status: Submitted  
Reviewer: uDJv

### Summary And Contributions
Multi-marginal flow matching learns a stochastic differential equation (SDE) that generates observed trajectories using score matching methods. This paper advances the literature by introducing a specific piecewise-quadratic conditional path and by learning the diffusion term of the SDE. The experiments are conducted primarily on clinical datasets related to brain imaging.

**Paper Keywords:** Generative modeling, disease progression, trajectory modeling  
**Expertise Keywords:** Sampling, disease progression, longitudinal data analysis

### Soundness
**Assessment:** Major errors (e.g., an incorrect theorem or derivation)

**Justification:**  
The proof of Proposition 3.1 is not rigorous, it's more an "intuitive" proof (assumptions lacking, or not clearly justified). I didn't find a rigorous proof in Appendix. Can you write a rigorous proof in Appendix ? (At least clarifying assumptions and explaining them)

### Significance
**Assessment:** Somewhat significant (e.g., theoretical contribution of limited novelty or empirical gains missing baselines/statistical rigor).

**Justification:**  
The novelty of the methodology is essentially in 3.2. Experiments focus on disease progression with appropriate comparisons and metrics. The proposed method outperforms the state-of-the-art.

However, I don't understand clearly the interest of the method for Alzheimer's disease progression compared to the state-of-the-art in longitudinal data analysis regarding forecasting. Can you develop more precisely the significance of your results ?

Why did you use ADNI-1 and not ADNI-III for the experiments ?

### Novelty
**Assessment:** Incremental compared to existing results

**Justification:**  
The method is explicit incremental on [1] and similar to [2]. The paper improve on the literature by using specific piecewise-quadratic conditional path and by learning the diffusion term of the SDE. The experiments focus on disease progression, which is also a specificity of the paper.

> [1] Tong, Alexander, et al. "Simulation-free schr" odinger bridges via score and flow matching." arXiv preprint arXiv:2307.03672 (2023).
>
> [2] Rohbeck, M., De Brouwer, E., Bunne, C., Huetter, J.- C., Biton, A., Chen, K. Y., Regev, A., and Lopez, R. (2025). Modeling complex system dynamics with flow matching across time and conditions. In The Thir- teenth Intern

### Non-conventional Contributions
No

### Clarity
The paper is mainly clearly written with extensive appendix.

### Relation To Prior Work
All related works are clearly discussed.

### Additional Comments
I tried to understand what you meant by “augmented empirical distributions” by looking at the reference you mentioned (Mok and Chung, 2020), but I couldn’t find these exact words using Ctrl+F. It was only by reading the appendix that I realized you had pre-processed your trajectories using diffeomorphic registration. Were you referring to spatial alignment, or did I misunderstand?

In future directions, you mentioned - « Finally, to improve plausibility and generalization in data-scarce settings, the model can be fortified with domain-specific knowledge. Integrating frameworks from scientific machine learning, such as Physics-Informed Neural Networks (PINNs), can constrain the learned dynamics to adhere to known governing equations »

I am not aware of any physic equations governing Alzheimer’s disease progression, do you ?

### Reproducibility
Sufficient amount of details available for reproducing the main results.

### Rating and Confidence
- **Rating:** 6: Accept (Technically solid paper, with high impact on at least one sub-area. The contribution is convincing and the evaluation is adequate, though there may be some limitations or open questions.)
- **Confidence:** 3: Fairly confident (It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.)

---

## Rebuttal by Authors (to Reviewer uDJv)

**Rebuttal by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*28 Nov 2025, 16:51 (modified: 01 Dec 2025, 14:11)*    
Status: Submitted

### Rebuttal

1. **On Proposition 3.1 and proof rigor.**

We thank the reviewer for pointing out the lack of rigor in our initial proof of Proposition 3.1. We agree that the original sketch was intuitive rather than formal and that the necessary assumptions were not explicitly stated.

In the revised manuscript, we have moved the proof to the Appendix (Appendix A.1) and rewrote it formally. We now explicitly state the three key assumptions required for the result:

- **Regularity:** Differentiability of the probability density, drift, and diffusion to allow interchange of expectation and differentiation.
- **Realizability:** The function class for the drift and score is expressive enough to contain the true probability flow drift and score.
- **Optimality:** We analyze the gradient at a global minimum \((\theta^\*, \phi^\*)\) of the SDE objective, where the drift and score are perfectly learned.

Under these assumptions, we formally show that the gradient of the uncertainty objective vanishes at \((\theta^\*, \phi^\*)\). This is because:

- The learned drift matches the true drift, implying the residual has zero conditional mean (Lemma 3.1).
- The diffusion term \(\sigma\) is trained to match the variance of this residual.

Consequently, both terms in the gradient of the uncertainty loss are zero at the optimum.

We have incorporated these changes in the revised paper (Section 3.3 and Appendix A.1). Specifically, we added a "Remark on Assumptions" paragraph in the Appendix to provide the intuition behind these conditions, as requested. Furthermore, we updated the main text following Proposition 3.1 to explicitly state the practical consequence of this result: it guarantees that the uncertainty objective does not bias the learning of the drift and score, justifying our joint optimization approach.

2. **On the significance for Alzheimer's disease progression and ADNI choice.**

We appreciate the reviewer's question regarding our focus on the significance of Alzheimer's disease (AD) progression and our use of the ADNI-1 dataset instead of ADNI-3.

We want to emphasize that with our forecasting method, we can potentially diagnose 8 out of 100 patients AD 18 months earlier. This is significant since "dementia is currently the seventh leading cause of death and one of the major causes of disability and dependency among older people globally" [1], and 60% of dementia cases are AD [2]. Early AD diagnoses benefit patients (seeking early treatment options) and carers (preparing for AD onset) [3], and have large economic benefits [4].

AD makes for an ideal setting for evaluating longitudinal generative forecasting models, since it has well-documented structural progression patterns in MRI. Our choice of ADNI-1 was driven by a practical trade-off between scale and computational feasibility. ADNI-3 is much larger and including it in addition to our other imaging datasets would have exceeded our compute budget. We think that our experiments are stronger when multiple forecasting tasks are considered instead of just one large task.

We also note that in many clinical applications, access to very large longitudinal imaging datasets is uncommon. Demonstrating that our method can operate on smaller-scale datasets such as ADNI-1 is therefore valuable. It shows that the approach is realistic and valid in settings where data is limited.

We now clarify these points that motivate the use of ADNI in Section 4 (Datasets) and elaborate upon the significance of early AD diagnosis in Section 6.

> [1] https://www.who.int/news-room/fact-sheets/detail/dementia
>
> [2] 2024 Alzheimer's disease facts and figures. Alzheimers Dement. 2024 May;20(5):3708-3821. doi: 10.1002/alz.13809. Epub 2024 Apr 30. PMID: 38689398; PMCID: PMC11095490.
>
> [3] Rasmussen, Jill, and Haya Langerman. "Alzheimer’s disease–why we need early diagnosis." Degenerative neurological and neuromuscular disease (2019): 123-130.
>
> [4] Leifer, Bennett P. "Early diagnosis of Alzheimer's disease: clinical and economic benefits." Journal of the American Geriatrics Society 51.5s2 (2003): S281-S288.

3. **On “augmented empirical distributions” and Mok & Chung (2020).**

We thank the reviewer for pointing out that the notion of "augmented empirical distribution" was not clearly defined in the manuscript. We agree and will clarify this in the revision.

In the context of our paper, what we intended to convey with this term is that when aligning longitudinal imaging data, one implicitly defines a distribution over possible alignments. This distribution consists of all possible alignment scenarios that can be affine, diffeomorphic, or other heuristic-based transformations. So the "augmented empirical distribution" refers to the dataset after it has been mapped through these alignment transformations, and is fully determined by the particular registration heuristic that was used. The reference to Mok & Chung (2020) was meant to contextualize the type of alignment we use: diffeomorphic registration methods that generate smooth and invertible spatial mappings between timepoints.

To avoid confusion, we updated Section 3.4 to be more explicit in the revised manuscript on what, in practice, we mean with the term "augmented empirical distributions".

4. **On PINNs and disease progression dynamics.**

We appreciate this important nuance. Our intention in the “future directions” paragraph was to point out that one could combine IMMFM with physics-informed regularization to further constrain the learned dynamics, in domains where mechanistic models are available. Indeed, for Alzheimer's disease specifically, no such universally accepted set of governing equations exists (yet!). Mechanistic and network-based models do exist, e.g., biomarker cascade hypotheses, network diffusion models of pathology spread. However, these are high-level and also often debated.

To avoid overstating the current maturity of mechanistic models in AD, we will rephrase this section to emphasize that PINN-style constraints are more applicable in domains with well-established governing equations. In addition, we will clarify that in the context of AD, our suggestion is more speculative and refers to the possibility of incorporating approximate mechanistic priors (such as network diffusion models) rather than strict physical laws.

---

## Official Comment by Reviewer uDJv (follow-up)

**Official Comment by Reviewer uDJv**  
*03 Dec 2025, 08:27*    
Status: Submitted

**Comment:**  
Regarding points 1. 2. and 4, I'm satisfied, thank you for your detailed answer.

Regarding points 3., I'm quite skeptical regarding your claim which is quite bold "We want to emphasize that with our forecasting method, we can potentially diagnose 8 out of 100 patients AD 18 months earlier."; compared to which baseline? generative models (as in your benchmark) or old-school longitudinal data analysis model ? From what I know, old-school models are still competitive (See [1] for example). I suggest to compare to Leaspy in order to check the generality of your claim.

> [1] Benchmarking parametric models of disease progression for early detection of cognitive decline

---

## Official Comment by Authors (reply to Reviewer uDJv)

**Official Comment by Authors** (Thijs P. Kuipers, Erik J Bekkers, Coen de Vente, Sharvaree Vadgama, +3 more)  
*03 Dec 2025, 18:09 (modified: 03 Dec 2025, 22:57)*    
Status: Submitted

**Comment:**  
Thank you for raising this concern. We see that our phrasing may have sounded too strong, overshadowing the message we intend to convey here. We want to clarify that we do not intend to imply that our conversion prediction results are state-of-the-art. Below, we provide some context on the meaning of our claim and why we find it important.

Our forecasting model is not specifically designed, tuned, or optimized for (AD) disease conversion prediction. What it is, is a generic high-dimensional longitudinal forecasting simulator. In contrast to dedicated parametric disease-progression models like Leaspy, our AD diagnosis is only based on the follow-up MRI image, so no biomarkers, longitudinal cognitive scores, or AD-specific features are used. Moreover, the downstream prediction we report is based on a simple threshold of observed shift in the ventricle area. Despite this setup, the forecasted images still allow us to correctly identify 8/100 future converters, which we find notable because of the following reasons:

- Our method uses only MRI images and has no prior knowledge about AD conversion as a downstream task during training.
- We use a simple threshold-based readout to isolate the effect of the progression simulation itself. A prediction model trained specifically for AD progression directly on the simulated data (or latents) and handpicked biomarkers that the parametric models use could potentially improve the results, but this is not the goal of our experiment.
- The goal is to see whether our simulations are meaningful. The fact that our prediction mechanism yields meaningful results suggests that the model is not just "morphing" images, but is capturing pathology-related geometric and structural trajectories relevant to AD.

In contrast, Leaspy instead directly models neuropsychological score, ADAS-Cog, which is also used as a measure of outcome in AD clinical trials. [1] This means that it directly operates on strong AD outcome measurements, making it highly non-generic.

Because of the above points, we view our method as adding complementary value rather than claiming superiority over the AD disease progression methods. But additionally, with our method, the forecasted images could potentially be used for further downstream analysis, for example, Voxel-Based Morphometry (VBM) to detect local gray matter density changes, perform cortical thickness analysis to map thinning patterns, or extract radiomic texture features to obtain a more detailed picture of the disease state. In future, our forecasting method and the parametric methods like Leaspy could be combined, for example, by using forecasted images to enrich biomarkers.

We hope that this additional context adds more clarity to our original claim. We also agree that this clarity is missing in the current manuscript, and we will discuss this more thoroughly in Section 5 (Discussion) in the revised manuscript.

> Wessels AM, Dowsett SA, Sims JR. Detecting Treatment Group Differences in Alzheimer's Disease Clinical Trials: A Comparison of Alzheimer's Disease Assessment Scale - Cognitive Subscale (ADAS-Cog) and the Clinical Dementia Rating - Sum of Boxes (CDR-SB). J Prev Alzheimers Dis. 2018;5(1):15-20. doi: 10.14283/jpad.2018.2. PMID: 29405227; PMCID: PMC12280810.

---

## Author Review Rating of Submission1678 by Mohammad Mohaiminul Islam (for Reviewer uDJv)

**Author Review Rating by** Mohammad Mohaiminul Islam  
*28 Dec 2025, 15:23*  eets expectations
