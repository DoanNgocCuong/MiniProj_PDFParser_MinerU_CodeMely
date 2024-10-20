# H OW TO  R OBUSTIFY  B LACK -B OX  ML M ODELS ? A Z EROTH -O RDER  O PTIMIZATION  P ERSPECTIVE  

Yimeng Zhang Yuguang Yao Jinghan Jia Michigan State University Michigan State University Michigan State University  

Jinfeng Yi Mingyi Hong Shiyu Chang Sijia Liu JD AI Research University of Minnesota UC Santa Barbara Michigan State University  

# A BSTRACT  

The lack of adversarial robustness has been recognized as an important issue for state-of-the-art machine learning (ML) models, e.g., deep neural networks (DNNs). Thereby, robustifying ML models against adversarial attacks is now a major fo- cus of research. However, nearly all existing defense methods, particularly for robust training, made the  white-box  assumption that the defender has the access to the details of an ML model (or its surrogate alternatives if available), e.g., its architectures and parameters. Beyond existing works, in this paper we aim to address the problem of  black-box defense : How to robustify a black-box model using just input queries and output feedback? Such a problem arises in practical scenarios, where the owner of the predictive model is reluctant to share model information in order to preserve privacy. To this end, we propose a general no- tion of defensive operation that can be applied to black-box models, and design it through the lens of denoised smoothing (DS), a ﬁrst-order (FO) certiﬁed de- fense technique. To allow the design of merely using model queries, we further integrate DS with the zeroth-order (gradient-free) optimization. However, a di- rect implementation of zeroth-order (ZO) optimization suffers a high variance of gradient estimates, and thus leads to ineffective defense. To tackle this problem, we next propose to prepend an autoencoder (AE) to a given (black-box) model so that DS can be trained using variance-reduced ZO optimization. We term the eventual defense as ZO-AE-DS. In practice, we empirically show that ZO-AE- DS can achieve improved accuracy, certiﬁed robustness, and query complexity over existing baselines. And the effectiveness of our approach is justiﬁed under both image classiﬁcation and image reconstruction tasks. Codes are available at https://github.com/damon-demon/Black-Box-Defense .  

# 1 I NTRODUCTION  

ML models, DNNs in particular, have achieved remarkable success owing to their superior predictive performance. However, they often lack robustness. For example, imperceptible but carefully-crafted input perturbations can fool the decision of a well-trained ML model. These input perturbations refer to  adversarial perturbations , and the adversarially perturbed (test-time) examples are known as  adversarial examples  or  adversarial attacks  (Goodfellow et al., 2015; Carlini & Wagner, 2017; Papernot et al., 2016). Existing studies have shown that it is not difﬁcult to generate adversarial attacks. Numerous attack generation methods have been designed and successfully applied to  (i) different use cases from the digital world to the physical world,  e.g. , image classiﬁcation (Brown et al., 2017; Li et al., 2019; Xu et al., 2019; Yuan et al., 2021), object detection/tracking (Eykholt et al., 2017; Xu et al., 2020; Sun et al., 2020), and image reconstruction (Antun et al., 2020; Raj et al., 2020; Vasiljevi´ c et al., 2021), and  (ii)  different types of victim models,  e.g. , white-box models whose details can be accessed by adversaries (Madry et al., 2018; Carlini & Wagner, 2017; Tramer et al., 2020; Croce & Hein, 2020; Wang et al., 2021), and black-box models whose information is not disclosed to adversaries (Papernot et al., 2017; Tu et al., 2019; Ilyas et al., 2018a; Liang et al., 2021).  

Problem statement. Let $f_{\mathbftheta_{\mathrm{bb}}}(\mathbf x)$  denote a pre-deﬁned  b lack- b ox $(b b)$  predictive model , which can map an input example $\mathbf{x}$  to a prediction. In our work,   $f_{\theta_{\mathrm{bb}}}$  can be either an image classiﬁer or an image reconstructor. For simplicity of notation, we will drop the model parameters $\theta_{\mathrm{bb}}$  when referring to a black-box model. The  threat model  of our interest is given by norm-ball constrained adversarial attacks (Goodfellow et al., 2015). To defend against these attacks, existing approaches commonly require the white-box assumption of   $f$  (Madry et al., 2018) or have access to white-box surrogate models of   $f$  (Salman et al., 2020). Different from the prior works, we study the problem of  black-box defense  when the owner of $f$  is not able to share the model details. Accordingly, the only mode of interaction with the black-box system is via submitting inputs and receiving the corresponding predicted outputs. The formal statement of black-box defense is given below:  

(Black-box defense)  Given a black-box base model $f$ , can we develop a defensive operation

 $\mathcal{R}$  using just input-output  function queries  so as to produce the robustiﬁed model   $\mathcal{R}(f)$  against adversarial attacks?  

Defensive operation. We next provide a concrete formul  of the defensive operation $\mathcal{R}$ . In the literature, two principled defensive operatio ere used: ( $(\mathcal{R}_{1})$ R ) end-to-end AT (Madry et al., 2018; Zhang et al., 2019b; Cohen et al., 2019), and ( $(\mathcal{R}_{2})$ R ) prepending a defensive compon  a base model (Meng & Chen, 2017; Salman et al., 2020; Aldahdooh et al., 2021). The former ( $(\mathcal{R}_{1})$ R ) has achieved the state-of-the-art robustness performance (Athalye et al., 2018; Croce & Hein, 2020) but is not applicable to black-box defense. By contrast, the latter  $(\mathcal{R}_{2})$  is more  patible with black-box models. For example,  denonised smoothing  (DS), a recently-developed  R -type approach (Salman et al., 2020), gives a certiﬁed defense by prepending a custom-trained denoiser to the targeted model. In this work, we choose DS as the backbone of our defensive operation (Fig. 2).  

In DS, a denoiser is integrated with a base model $f$  so that the augmented system becomes resilient to Gaussian noise and thus plays a role similar to the RS-based certiﬁed defense (Cohen et al., 2019). That is, DS yields  

$$
{\mathcal R}(f({\bf x})):=f(D_{\theta}({\bf x})),
$$  

where $D_{\theta}$  denotes the learnable denoiser (with  

![](images/a3e5a6b3c8346a94b0672fca128f1674de9efb5b785b01a67c219a399b338c1b.jpg)  
Figure 2:  DS-based black-box defense.  

parameters $\pmb{\theta}$ ) prepended to the (black-box) predictor   $f$ . Once   $D_{\theta}$  is learned, then the DS-based smooth classiﬁer,  arg max $\mathbb{P}_{\pmb{\delta}\in\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})}[\mathcal{R}(f(\mathbf{\bar{x}}+\pmb{\delta}))\overset{\cdot}{=}c]$ , can achieve certiﬁed robustnes where c $c$ $\delta\,\in\,\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})$ denotes the standard Gaussia  noise with variance  σ $\sigma^{2}$ , and $\begin{array}{r}{\mathrm{reg}\operatorname*{max}_{c}\mathbb{P}_{\delta\in\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})}[f(\mathbf{x}+\delta)=c]}\end{array}$  signiﬁes a smooth version of $f$ . ∈N  

Based on (1), the goal of black-box defense becomes to ﬁnd the optimal denoiser   $D_{\theta}$  so as to achieve satisfactory accuracy as well as adversarial robustness. In the FO learning paradigm, Salman et al. (2020) proposed a stability regularized denoising loss to train $D_{\theta}$ :  

$$
\begin{array}{r}{\underset{\theta}{\mathrm{minimize}}\;\;\mathbb{E}_{\delta\in\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I}),\mathbf{x}\in\mathcal{U}}\underbrace{\|D_{\theta}(\mathbf{x}+\delta)-\mathbf{x}\|_{2}^{2}}_{:=\,\ell_{\mathrm{Denso}}(\theta)}+\gamma\mathbb{E}_{\delta,\mathbf{x}}\underbrace{\ell_{\mathrm{CE}}(\mathcal{R}(f(\mathbf{x}+\delta)),f(\mathbf{x}))}_{:=\,\ell_{\mathrm{Stab}}(\theta)},}\end{array}
$$  

where $\mathcal{U}$  denotes the training dataset, the ﬁrst objective term $\ell_{\mathrm{QEDOISe}}(\pmb{\theta})$ orresponds to the mean squared error (MSE) of image denoising, the second objective term $\ell_{\mathrm{Stab}}(\pmb\theta)$  measures the prediction stability through the cross-entropy (CE) between the outputs of the denoised input and the original input, and $\gamma>0$  is a regularization parameter that strikes a balance between   $\ell_{\mathrm{Denominus}}$  and   $\ell_{\mathrm{Stab}}$ .  

We remark that problem (2) can be solved using the FO gradient descent method if the base model $f$  is fully disclosed to the defender. However, the black-box nature of   $f$  makes the gradients of the stability loss   $\ell_{\mathrm{Stab}}(\pmb\theta)$  infeasible to obtain. Thus, we will develop a  gradient-free  DS-oriented defense.  

# 4 M ETHOD : A S CALABLE  Z EROTH -O RDER  O PTIMIZATION  S OLUTION  

In this section, we begin by presenting a brief background on ZO optimization, and elaborate on the challenge of black-box defense in high dimensions. Next, we propose a novel ZO optimization-based DS method that can not only improve model query complexity but also lead to certiﬁed robustness.  

Baselines. We will consider two  variants  of our proposed ZO-AE-DS:  i) ZO-AE-DS using RGE (3),  ii) ZO-AE-DS using CGE  (4). In addition, we will compare ZO-AE-DS with  i) FO-AE-DS , i.e. , the ﬁrst-order implementation of ZO-AE-DS,  ii) FO-DS , which developed in (Salman et al., 2020),  iii) RS -based certiﬁed training, proposed in (Cohen et al., 2019), and $i\nu$ ) ZO-DS ,  i.e. , the ZO implementation of FO-DS using RGE. Note that CGE is not applicable to ZO-DS due to the obstacle of high dimensions. To our best knowledge, ZO-DS is the only query-based black-box defense baseline that can be directly compared with ZO-AE-DS.  

Training setup. We build the training pipeline of the proposed ZO-AE-DS following ‘Training ZO-AE-DS’ in Sec. 4. To optimize the denoising model   $D_{\theta}$ , we will cover two training schemes: training from scratch, and pre-training & ﬁne-tuning. In the scenario of training from scratch, we use Adam optimizer with learning rate   $10^{-3}$   to train the model for 200 epochs and then use SGD optimizer with learning rate $10^{-3}$   drop by a factor of  10  at every 200 epoch, where the total number of epochs is 600. As will be evident later, training from scratch over   $D_{\theta}$  leads to better performance of ZO-AE-DS. In the scenario of pre-training  $\&$  ﬁne-tuning, we use Adam optimizer to pre-train the denoiser $D_{\theta}$  with the MSE loss $\ell_{\mathrm{Denominus}}$  in (2) for 90 epochs and ﬁne-tune the denoiser with $\ell_{\mathrm{Stab}}$  for 200  epochs with learning rate $10^{-5}$   drop by a factor of  10  every 40 epochs. When implementing the baseline FO-DS, we use the best training setup provided by (Salman et al., 2020). When implementing ZO-DS, we reduce the initial learning rate to   $\bar{10}^{-4}$   for training from scratch and $10^{-6}$   for pre-training & ﬁne-tuning to stabilize the convergence of ZO optimization. Furthermore, we set the smoothing parameter   $\mu=0.005$  for RGE and CGE. And to achieve a smooth predictor, we set the Gaussian smoothing noise as   $\pmb{\delta}\in\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})$  with   $\sigma^{2}\,=\,0.25$ . With the help of matrix operations and the parallel computing power of the GPU, we optimize the training time to an acceptable range. The averaged one-epoch training time on a single Nvidia RTX A60  is about $\sim1\mathrm{{min}}$  and   $\sim29\mathrm{min}$ for FO-DS and our proposed ZO method, ZO-AE-DS (CGE, $q=192$ ), on the CIFAR-10 dataset.  

Evaluation metrics. In the task of robust image classiﬁcation, the performance will be evaluated at s tandard test  a ccuracy (SA) and  c ertiﬁed  a ccuracy (CA). Here CA is a provable robust guarantee of the Gaussian smoothing version of a predictive model. Let us take ZO-AE-DS as an example, the resulting smoot ge classiﬁer is given by $f_{\mathrm{smooth}}(\mathbf{x}):=\,\mathrm{arg\,max}_{c}\,\mathbb{P}_{\delta\in\mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})}[\mathcal{R}_{\mathrm{new}}(f(\mathbf{x}+\delta))=c].$ , where  R ${\mathcal{R}}_{\mathrm{new}}$  is given by (6). Further, a  certiﬁed radius  of $\ell_{2}$ -norm perturbation ball with respect to an input example can be calculated following the RS approach provided in (Cohen et al., 2019). As a result, CA at a given $\ell_{2}$ -radius $r$  is the percentage of the correctly classiﬁed data points whose certiﬁed radii are larger than $r$ . Note that if   $r=0$ , then CA reduces to SA.  

![](images/8a5eef83f12490c35e15c0e686d7a57042fea8c036116bd851b5342481f3d67f.jpg)  

Table 2:  SA (standard accuracy, $\%$ ) and CA (certiﬁed accuracy, $\%$ ) versus different values of   $\ell_{2}$ -radius $r$ . Note that SA corresponds to the case of $r=0$ . In both FO and ZO blocks, the best accuracies for each   $\ell_{2}$ -radius are highlighted in  bold .  

# 5.2 E XPERIMENT RESULTS ON IMAGE CLASSIFICATION  

Performance on CIFAR-10. In Table 2, we present certiﬁed accuracies of ZO-AE-DS and its variants/baselines versus different   $\ell_{2}$ -radii in the setup of (CIFAR-10, ResNet-110). Towards a comprehensive comparison, different RGE-based variants of ZO-AE-DS and ZO-DS are demonstrated using the query number   $q\in\{20,100,192\}$ . First, the comparison between ZO-AE-DS and  shows that our proposal signiﬁcantly outperforms ZO-DS ranging from the low query number $q=20$ to the high query number   $q=192$  when RGE is applied. Second, we observe that the use of CGE yields the best CA and SA (corresponding to   $r=0$ ). The application of CGE is beneﬁted from AE, which reduces the dimension from   $d=32\times32\times3$  to   $d_{z}=192$ . I particul -based ZO-AE-DS improves the case studied in Table 1 from $5.06\%$  to $35.5\%$  at the $\ell_{2}$ -radius $r=0.5$ . Third, although FO-AE-DS yields CA improvement over FO-DS in the white-box context, the improvement achieved by ZO-AE-DS (vs. ZO-DS) for black-box defense is much more signiﬁcant. This implies Following (Antun et al., 2020; Raj et al., 2020), we generate the noisy measurement following a linear observation model $\mathbf{y}=\mathbf{Ax}$ , where  A  is a sub-sampling matrix ( e.g. , Gaussian sampling), and $\mathbf{x}$  is an original image. A pre-trained image reconstruction network (Raj et al., 2020) then takes   $\mathbf{A}^{\top}\mathbf{y}$  as the input to recover  x . To evaluate the reconstruction performance, we adopt two metrics (Antun et al., 2020), the root mean squared error (RMSE) and structural similarity (SSIM). SSIM is a supplementary metric to RMSE, since it gives an accuracy indicator when evaluating the similarity between the true image and its estimate at ﬁne-level regions. The vulnerability of image reconstruction networks to adversarial attacks,  e.g. , PGD attacks (Madry et al., 2018), has been shown in (Antun et al., 2020; Raj et al., 2020; Wolf, 2019).  

When the image reconstructor is given as a black-box model, spurred by above, Table 5 presents the performance of image reconstruction using various training methods against adversarial attacks with different perturbation strengths. As we can see, compared to the normally trained image reconstructor ( i.e. , ‘Standard’ in Table 5), all robustiﬁcation methods lead to degraded standard image reconstruction performance in the non-adversarial context ( i.e. ,   $\|\pmb{\delta}\|_{2}=0$ ). But the worst performance is provided by ZO-DS. When the perturbation strength increases, the model achieved by standard training becomes over-sensitive to adversarial perturbations, yielding the highest RMSE and the lowest SSIM. Furthermore, we observe that the proposed black-box defense ZO-AE-DS yields very competitive and even better performance with respect to FO defenses. In Fig. 5, we provide visualizations of the reconstructed images using different approaches at the presence of reconstruction-evasion PGD attacks. For example, the comparison between Fig. 5-(f) and (b)/(d) clearly shows the robustness gained by ZO-AE-DS.  

Table 5: Performance of image reconstruction using different methods at various attack scenarios. Here ‘standard’ refers to the original image reconstructor without making any robustiﬁcation. Four robustiﬁcation methods are presented including FO-DS, ZO-DS (RGE, $q=192$ ), FO-AE-DS, and ZO-AE-DS (CGE,   $q=192$ ). The performance metrics RMSE and SSIM are measured by adversarial example   $({\bf x}+{\pmb\delta})$ , generated by  40 -step $\ell_{2}$  PGD attacks under different values of $\ell_{2}$  perturbation norm   $||\pmb{\delta}||_{2}$ . 
![](images/7acce215b06933d14db02486aeaf914897995b3fc935a24b2b1e073f14f44ab3.jpg)  

![](images/3fa3a90f308e24dbaf7a2096e10d6ae6582c1a616b40f18f564ef1dbce4f4a1c.jpg)  
Figure 5:  Visualization for Image Reconstruction under   $\ell_{2}$  PGD attack (Step $=40$ , $\epsilon=1.0$  ). Original: base reconstruction network. ZO-DS: RGE with $q=192$ . ZO-AE-DS: CGE with $q=192$  

# 6 C ONCLUSION  

In this paper, we study the problem of black-box defense, aiming to secure black-box models against adversarial attacks using only input-output model queries. The proposed black-box learning paradigm is new to adversarial defense, but is also challenging to tackle because of the black-box optimization nature. To solve this problem, we integrate denoised smoothing (DS) with ZO (zeroth- order) optimization to build a feasible black-box defense framework. However, we ﬁnd that the direct application of ZO optimization makes the defense ineffective and difﬁcult to scale. We then propose ZO-AE-DS, which leverages autoencoder (AE) to bridge the gap between FO and ZO optimization. We show that ZO-AE-DS reduces the variance of ZO gradient estimates and improves the defense and optimization performance in a signiﬁcant manner. Lastly, we evaluate the superiority of our proposal to a series of baselines in both image classiﬁcation and image reconstruction tasks.  

# A D ERIVATION OF  (8)  

First, based on (2) and (6), the stability loss corresponding to ZO-AE-DS is given by  

$$
f^{\prime}(\mathbf{z})=f\left(\phi_{\pmb\theta_{\mathrm{Dec}}}\left(\mathbf{z}\right)\right),\ \ \mathbf{z}=\psi_{\pmb\theta_{\mathrm{Enc}}}\left(D_{\pmb\theta}(\mathbf{x}+\pmb\delta)\right).
$$
 $\ell_{\mathrm{Stab}}(\pmb{\theta})=\ell_{\mathrm{CE}}\left(f^{\prime}(\mathbf{z}),f(\mathbf{x})\right):=g(\mathbf{z}).$ ,  where  

We then take the derivative of $\ell_{\mathrm{Stab}}(\pmb\theta)$  w.r.t.   $\pmb{\theta}$ . This yields  

$$
\nabla_{\pmb\theta}\ell_{\mathrm{Stab}}(\pmb\theta)=\frac{d\mathbf z}{d\pmb\theta}\frac{d g(\mathbf z)}{d\mathbf z}\mid_{\mathbf z=\psi_{\pmb\theta_{\mathrm{Enc}}}(D_{\pmb\theta}(\mathbf x+\pmb\delta))},
$$  

where   $\begin{array}{r}{\frac{d\mathbf{z}}{d\pmb{\theta}}\in\mathbb{R}^{d_{\theta}\times d}}\end{array}$  and $\begin{array}{r}{\frac{d g(\mathbf{z})}{d\mathbf{z}}\in\mathbb{R}^{d}}\end{array}$ .  

Since   $g(\mathbf{z})$  involves the black-box function   $f$ , we ﬁrst compute its ZO gradient estimate following (3) or (4) and obtain  

$$
\frac{d g(\mathbf{z})}{d\mathbf{z}}\mid_{\mathbf{z}=\psi_{\theta_{\mathrm{Enc}}}(D_{\theta}(\mathbf{x}+\delta))}\approx\hat{\nabla}_{\mathbf{z}}g(\mathbf{z})\mid_{\mathbf{z}=\psi_{\theta_{\mathrm{Enc}}}(D_{\theta}(\mathbf{x}+\delta))}:=\mathbf{a}.
$$  

Substituting the above into (10), we obtain  

$$
\nabla_{\pmb\theta}\ell_{\mathrm{Stab}}(\pmb\theta)=\frac{d\mathbf z}{d\pmb\theta}\mathbf a=\left[\begin{array}{c}{\frac{d\mathbf a^{\top}\mathbf z}{d\theta_{1}}}\\ {\frac{d\mathbf a^{\top}\mathbf z}{d\theta_{2}}}\\ {\vdots}\\ {\frac{d\mathbf a^{\top}\mathbf z}{d\theta_{d_{\theta}}}}\end{array}\right]=\frac{d\mathbf a^{\top}\mathbf z}{d\pmb\theta}=\nabla_{\pmb\theta}[\mathbf a^{\top}\phi_{\theta_{\mathrm{Enc}}}(D_{\pmb\theta}(\mathbf x+\pmb\delta))],
$$  

where the last equality holds based on (9). This completes the derivation.  

# B C OMBINATION OF  D IFFERENT  D ENOISERS AND  C LASSIFIERS  

Table A1 presents the certiﬁed accuracies of our proposal using different denoiser models (Wide- DnCnn vs. DnCnn) and image classiﬁer  $(\mathrm{Vgg{-}16})$ .  

Table A1:  CA (certiﬁed accuracy, $\%$ ) vs. different $\ell_{2}$ -radii for different combinations of denoisers and classiﬁer. 
![](images/f033194867cfdba22dd5b3588359838d484b7c3c26ce400cf5e00a7101ae49f4.jpg)  

# C A DDITIONAL EXPERIMENTS AND ABLATION STUDIES  

In what follows, we will show the ablation study on the choice of AE architectures in Appendix C.1. Afterwards, we will show the performance of FO-AE-DS versus different training schemes in Appendix C.2. Finally, we will show the performance of our proposal on the high-dimension ImageNet images in Appendix C.3.  

# C.1 T HE PERFORMANCE OF  FO-AE-DS  WITH DIFFERENT  A UTO E NCODERS .  

Table. A2 presents the certiﬁed accuracy performance of FO-AE-DS with different autoencoders (AE). As we can see, if AE-96 is used (namely, the encoded dimension is half of AE-192 used in the paper), then we observe a slight performance drop. This is a promising result as we can further reduce the query complexity by choosing a different autoencoder since the use of CGE has to be matched with the encoded dimension.  

Table A2:  CA (certiﬁed accuracy,  $\%$ ) vs. different $\ell_{2}$ -radii for FO-AE-DS with different AutoEncoders. 
![](images/af3e86684807e788c1f9eb9fc2bd50625c22adb454c4699932289b025d3a6757.jpg)  

# C.2 T HE PERFORMANCE OF  FO-AE-DS  WITH DIFFERENT TRAINING SCHEMES  

Table. A3 presents the certiﬁed accuracy of FO-AE-DS (ﬁrst-order implementation of ZO-AE-DS) with different training schemes. Training both denoiser and encoder is the default setting. As we can see, only training the denoiser would bring performance degradation, and training both denoiser and AE does boost the performance. It is worth noting that FO-AE-DS with "train the denoiser and AE" training scheme can be regarded as the FO-DS treating the combination of the original denoiser and the same AE used in FO-AE-DS as a new denoiser, which cannot be implemented for ZO-AE-DS since the decoder of ZO-AE-DS is merged into the black-box classiﬁer and its parameters cannot be updated. Furthermore, the key of the introduced AE is to reduce the variable dimension for Zeroth-Order (ZO) gradient estimation.  

Table A3:  CA (certiﬁed accuracy, $\%$ ) vs. different   $\ell_{2}$ -radii for FO-AE-DS with different training schemes. 
![](images/873f17e6eb075efc55c0316150c816ac9554f358b71ae93c52c297e5bd85a52a.jpg)  

# C.3 T HE PERFORMANCE OF  ZO-AE-DS  ON  I MAGE N ET  I MAGES  

To evaluate the performance of ZO-AE-DS on the Restricted ImageNet (R-ImageNet) dataset, a 10-class subset of ImageNet with 38472 images for training and 1500 images for testing, similar to (Tsipras et al., 2019). Due to our limited computing resources, we are not able to scale up our experiment to the full ImageNet dataset, but the purpose of evaluating on high-dimension images remains the same. In the implementation of ZO-AE-DS, we choose an AE with an aggressive compression (130:1), which is to compress the original $3\times224\times224$  images into the   $1152\times1\times1$ feature dimension. We compare the certiﬁed accuracy (CA) performance of our proposed ZO-AE-DS (using CGE) with the black-box baseline ZO-DS, and the white-box baselines FO-DS and FO-AE-DS. Results are summarized in the following table.  

As we can see, (1) when considering the black-box classiﬁer, the proposed ZO-AE-DS still signif- icantly outperforms the direct ZO implementation of DS. This shows the importance of variance reduction of query-based gradient estimates. (2) Since ZO-AE-DS and FO-AE-DS used an aggressive AE structure, the performance drops compared to FO-DS. (3) the use of high-resolution images would make the black-box defense much more challenging. However, ZO-AE-DS is still a principled black-box defense method that can achieve reasonable performance.  

Table A4:  CA (certiﬁed accuracy,  $\%$ ) vs. different $\ell_{2}$ -radii for FO-AE-DS on ImageNet Images. 
![](images/aa70b7898617187709cea64e83e7be236ea15ea8f00305bec5df566b96321592.jpg)  