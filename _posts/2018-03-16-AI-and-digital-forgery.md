---
title: 'Commoditisation of AI, digital forgery and the end of trust: how we can fix it'
summary: 'With some experiments on detecting face swaps by humans and the machine'
layout: post
date: 2018-03-17
permalink: /posts/2018/03/17/AI-and-digital-forgery/
tags:
  - machine Learning
  - security
  - society
img-folder: 2018-03-17-AI-and-digital-forgery
---

<div style="text-align: right">
written with Simone Lini, Hamish Ivey-Law and
<a href="https://mortendahl.github.io">Morten Dahl</a>.
</div>

<br><br>

*TLDR; It is becoming widely evident that technology will enable total manipulation of video and audio content, as well as its digital creation from scratch. As a consequence, the meaning of evidence and trust will be critically challenged and pillars of the modern society such as information, justice and democracy will be shaken up and go through a period of crisis. Once tools for fabrication  becomes a commodity, the effects will be more dramatic than the current phenomenon of fake news. In the tech circles the issue are discussed only at a philosophical level; no clear solution is known at present time. This post discusses pros and cons of two classes of potential solutions: digital signatures and learning based detection systems. We also ran a brief "weekend experiment" to measure the effectiveness of machine learning for detection of face manipulation, on the wave of [deepfakes](https://motherboard.vice.com/en_us/topic/deepfakes). In the limited scope of the experiment, our model is able to spot image manipulation that is imperceptible to the human eye.*

<br>

**In 1983,** at the peak of the Cold War, [Stanislav Petrov](https://en.wikipedia.org/wiki/Stanislav_Petrov) was a lieutenant colonel stationed at the Serpukhov-15 bunker, close to the Soviet capital. He was monitoring the early warning system which was in charge of detecting nuclear missile launches from the United States.


In September 26, the bunker systems alerted Petrov of a missile launch from Montana.
The US and the USSR were following the *mutual assured destruction* doctrine. If the Americans were to launch a nuclear attack, the Soviets would have retaliated with a massive counterattack, ensuring the annihilation of both countries.

<center>
<p><a href="https://commons.wikimedia.org/wiki/File:Stanislaw-jewgrafowitsch-petrow-2016.jpg#/media/File:Stanislaw-jewgrafowitsch-petrow-2016.jpg"><img src="https://upload.wikimedia.org/wikipedia/commons/6/67/Stanislaw-jewgrafowitsch-petrow-2016.jpg" alt="Stanislaw-jewgrafowitsch-petrow-2016.jpg"></a>By <a href="//commons.wikimedia.org/w/index.php?title=User:Queery-54&amp;action=edit&amp;redlink=1" class="new" title="User:Queery-54 (page does not exist)">Queery-54</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=62394026">Link</a></p>
</center>

Had Petrov alerted his superiors, he would have started World War III. Instead, Petrov correctly recognised that it was unlikely for the US to attack by launching only five missiles, as he was seeing. It was a bug. Petrov reported it as a computer malfunction, instead of an attack. He was right. World War III was avoided thanks to the critical judgment of a man.

Now, let's make a thought experiment. A few years in the future, say in 2020, the situation between North Korea and the US is still tense. CNN receives a video from an anonymous source. Kim Jong-un appears in it, discussing in a secured facility with his generals. The footage was never seen before.

![](https://i.imgur.com/GJQsgJj.jpg)

Korean interpreters are called. The Supreme Leader is demanding the launch of a nuclear missile strike on the imminent [Day of the Sun](https://en.wikipedia.org/wiki/Public_holidays_in_North_Korea). In a matter of hours, the video gets to the Oval Office. Intelligence cannot confirm or deny the authenticity of the video, even by consulting additional sources. The US president must act. He orders a preventive attack. The war starts.

But was the video evidence enough to justify such decision?

<br><br>

**Machine learning in early 2018.** The issue is that, in a few years time, digital content may be heavily manipulated and at the same time so accurate to be indistinguishable from reality to human eyes and ears. Facial expressions can be [crafted ad-hoc](https://www.youtube.com/watch?v=ohmajJTcpNk&t=), [real people voices](https://lyrebird.ai/demo/) and [lip movements](https://www.youtube.com/watch?v=9Yq67CjDqvw) in a video can be adapted to follow a script. The base video itself might be from a real recording, but the meaning intended to convey could be dictated at wish.

Machine learning is *the thing* that is singularly most responsible. Progress on [generative models](https://blog.openai.com/generative-models/) owes to scientific breakthroughs from the last 5 years or so, one of which is the [generative adversarial network](https://www.technologyreview.com/s/610253/the-ganfather-the-man-whos-given-machines-the-gift-of-imagination/), or GAN. The core idea of GANs is learning a generative model for images by fooling an opponent detector model, which job is to distinguish between real and fake (generated) content; realism is build in as an objective function. And the whole field is moving fast from the inceptions of those ideas:

<blockquote class="twitter-tweet tw-align-center" data-lang="en"><p lang="en" dir="ltr">4 years of GAN progress (source: <a href="https://t.co/hlxW3NnTJP">https://t.co/hlxW3NnTJP</a> ) <a href="https://t.co/kmK5zikayV">pic.twitter.com/kmK5zikayV</a></p>&mdash; Ian Goodfellow (@goodfellow_ian) <a href="https://twitter.com/goodfellow_ian/status/969776035649675265?ref_src=twsrc%5Etfw">March 3, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

This technological trend will change, and *is changing* [many industries](https://www.weforum.org/whitepapers/creative-disruption-the-impact-of-emerging-technologies-on-the-creative-economy) which business revolves around digital media, such as advertisement, [fashion](https://qz.com/1090267/artificial-intelligence-can-now-show-you-how-those-pants-will-fit/), cinema, [design/manufacturing](https://arxiv.org/abs/1703.05192), as well [as](http://mario-klingemann.tumblr.com) [art](http://www.samim.io).

An impressively realistic demonstration of GANs was presented recently by [NVIDIA](https://www.youtube.com/watch?v=G06dEcZ-QTg); but many other learning and vision technical advances power the current progress. For example, the key idea behind deepfakes itself is the more traditional model of auto-encoders.

But how about stuff like Photoshop and CGI? Generating high quality fake photos (for example) has been possible for a long time with Photoshop. And, given [what can be achieved today with computer graphics](https://www.youtube.com/watch?v=OUIHzanm5Mk), what's the difference? Why should we be more threatened by these new developments than we are by a technologies that have been around for decades?

Two answers:

- Lower editing effort and necessary technical expertise. The extremely fast progress that the machine learning and vision communities have experienced in the last few years is also attributed to cheap computation by GPUs and cloud services, the availability of much video/audio/textual data for research and widely available code on the Internet. These same elements, all put together, are tearing down the ingress barriers for playing with deep learning tools as well as promising a rather flat learning curve for newcomers.
- Higher realism that goes beyond what can be achieved through more traditional computer graphics techniques, in particular when we talk about video and audio. The question of how to produce realistic media is delegated to algorithms that must figure it out by comparing with the look of real media in the training set.

In a scenario like in the above introduction, a government will put the video material through heavy expert scrutiny and seek to cross-check it via different means. However, what happens when the tools for generation of realistic content become a commodity and it is instead the typical citizens to be challenged with determining authenticity, while scrolling their Facebook feed?



<br><br>

**A speculative look at potential implications.**

<blockquote class="twitter-tweet tw-align-center" data-lang="en">
<p lang="en" dir="ltr">
The biggest casualty to AI won&#39;t be jobs, but the final and complete eradication of trust in anything you see or hear. <a href="https://t.co/sg9o4v2Q3f">https://t.co/sg9o4v2Q3f</a> <a href="https://t.co/nkj007LtEF">pic.twitter.com/nkj007LtEF</a></p>&mdash; Oli Franklin-Wallis (@olifranklin) <a href="https://twitter.com/olifranklin/status/937660128974852096?ref_src=twsrc%5Etfw">December 4, 2017</a>
</blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

We're used to trusting videos. If we see U.S. President giving a speech on CNN, we take for granted that those words you hear come from his mouth. In a future world where anyone can make the POTUS say anything, this assumption is not only wrong -- it's dangerous if that someone has a bad agenda.

You won't be able to trust CNN -- or any media broadcaster you consider reputable -- if you can't rely on their judgment for discerning real and fake sources of information. At the end of the day, **the trust we have in digital content is simply due to the absence of a technology capable of turning its meaning upside down by manipulation**.

Think now at that future when potentially every teenager from their basement could, by reading an online tutorial, run some code on their powerful electronic devices and generate realistic movie clips with "real people" acting in it. This is hardly far fetched, given the current technological trends. But before reaching such wide-spread consumption, surely enough, technology for forgery will be in the hands of state-level actors, and the chance of systematic abuse for whatever local or international agenda is real.

The end of trust is much more serious than fake news, as we define the phenomenon today. Fake news can be fact checked. Readers can make a conscious decision on whether to trust a source. When it comes to video manipulation, you can't even trust the most reputable and well-meaning newspaper, website or TV channel, not because they may want to manipulate the truth -- because they are no more likely to recognise a fake video than you are.

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/kayla-velasquez-199343-unsplash.jpg)

While the media is maybe an intuitive example, it goes much further than that. Take [deepfakes for pornography](https://motherboard.vice.com/en_us/topic/deepfakes). *Today*, anyone can take photos of a Hollywood celebrity, or of a known person, and put his/her face on pornographic content. Online tutorials can guide step by step in obtaining fairly realistic outcomes. Probably far from the intended use from its creator, technology like deepfakes can been used to make videos for [revenge porn](https://www.forbes.com/forbes/welcome/?toURL=https://www.forbes.com/sites/ianmorris/2018/02/05/fakeapp-allows-anyone-to-make-deepfake-porn-of-anyone/&refURL=https://www.google.nl/&referrer=https://www.google.nl/). Porn websites are [struggling to stop it](https://mashable.com/2018/02/12/pornhub-deepfakes-ban-not-working/#Wk6bXfz30PqE). Dramatically, this is a powerful instrument for blackmail *even if the fact did not happen*; once a compromising video is out, the cost of fixing reputation and convincing the public that it was a hoax is very high.

If no countermeasure is taken, justice and law enforcement will be much challenged by the end of trust. Consider a police investigation successfully tracing a phone call in which somebody gives away important details about a criminal act, essentially bringing strong evidence against him/herself. The voice recorded belongs to one of the police suspects. This is unquestionable for human ears. But how can the evidence be brought to court if the authorities cannot be sure it is authentic, not even after forensic analysis? There is no answer right now.

The production of high quality fake photos fake evidence can lead to a reversal of a fundamental tenet of our judicial system, that *people are innocent until proven guilty*. The subject of any controversial photo is called upon to justify themselves (perhaps provide an alibi) every time; the "superficial legitimacy" of the photographic evidence leading to an assumption of guilty until proven innocent. This becomes an unending task for the accused when generating fake controversial photos is free, which could end up as a kind of [DDoS attack](https://en.wikipedia.org/wiki/Denial-of-service_attack) on a person, as well as on the judicial personnel examining those cases.

High quality voice synthesis opens a Pandora's box for what concerns [social engineering](https://en.wikipedia.org/wiki/Social_engineering_(security)). When, [soon enough](http://research.baidu.com/neural-voice-cloning-samples/), we will be able to copycat voices given just a few audio samples of the victim, phone conversations will be routinely hacked. It is hard to imagine an area of society that won't endangered by the commoditisation of tools for [identity thefts](https://en.wikipedia.org/wiki/Identity_theft). Without additional authentication measures, criminals may even impersonate law enforcement officials, give orders or misleading information to subordinates, and disrupt the  interventions of the authority at the operational level.

Lawfareblog [has got you covered](https://lawfareblog.com/deep-fakes-looming-crisis-national-security-democracy-and-privacy) with a few more scenarios:

> - Fake videos could feature public officials taking bribes, uttering racial epithets, or engaging in adultery.
> - Soldiers could be shown murdering innocent civilians in a war zone, precipitating waves of violence and even strategic harms to a war effort.
> - A fake video might portray an Israeli official doing or saying something so inflammatory as to cause riots in neighboring countries, potentially disrupting diplomatic ties or even motivating a wave of violence.
> - False audio might convincingly depict U.S. officials privately “admitting” a plan to commit this or that outrage overseas, exquisitely timed to disrupt an important diplomatic initiative.
> - A fake video might depict emergency officials “announcing” an impending missile strike on Los Angeles or an emergent pandemic in New York, provoking panic and worse.

None of this is good news for democracy. Prepare for more and more political debates discussing news events fabricated out of thin air. Information is a building block of the democratic society, the effective counter-balance of the other [three powers](https://en.wikipedia.org/wiki/Separation_of_powers). Without trustworthy sources of information, the institution of democracy is challenged at its foundation and reduced to  an empty shell of formal declarations. Those premises paint a dark future, darker than what pessimists may consider the present with regard to lack of freedom of information, state controlled media and fake news.

<br><br>

**Is the picture really that dark?** As humans, our innate "lack of trust" works as immune system. We will need to look at videos the way Petrov looked at his radar screen. If you are an optimist, this argument may be enough to convince you that not all is lost. Once people become aware that a video recording is not necessarily a trustworthy testimony of facts, we can expect everyone to judge digital media with systematic suspicion. This may sound similar to how we are slowly adapting to filter out fake news.

History has a habit of repeating itself. Before printing became a commodity, a message printed on a poster/paper/journal was regarded as coming from a reputable source, e.g. the government or an publishing company. Today it makes no sense to say "if printed, it must be trustworthy" or not even "official", in a weaker sense. This cultural shift will happen as well for more sophistical media vehicles of information, such as videos, **which we currently trust because they cannot be easily and fully manipulated**. (The printing comparison came from somewhere lost on the Internet; let us know if you have a link to that article.)

If you are a pessimist, the argument won't convince you. And there are at least two good reasons against it. The first one is about [confirmation bias](https://en.wikipedia.org/wiki/Confirmation_bias): we are more inclined to look for and trust information confirming our own subjective prior beliefs. Fabricated videos may be easily taken for true if they are aligned with our (maybe wrong) personal views.

There is a second, more subtle, pessimist argument, even if we turn to be more suspicious about digital media. General lack of trust in media could be very harmful for the functioning of our society, as highlighted in [this commentary](https://lawfareblog.com/deep-fakes-looming-crisis-national-security-democracy-and-privacy). If we all adapt to be extremely skeptical by default, we will stop believing in any occurrence of unlikely events. Hiding unconventional and criminal behaviour could become a matter of dismissing facts as too absurd to be true:

<blockquote class="twitter-tweet tw-align-center" data-lang="en">
<p lang="en" dir="ltr">“I’ve been skeptical about the collusion and obstruction claims for the last year. I just don’t see the evidence....in terms of the collusion, it’s all a bit implausible based on the evidence we have.” Jonathan Turley on <a href="https://twitter.com/FoxNews?ref_src=twsrc%5Etfw">@FoxNews</a></p>&mdash; Donald J. Trump (@realDonaldTrump) <a href="https://twitter.com/realDonaldTrump/status/968462966864609280?ref_src=twsrc%5Etfw">February 27, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Do you remember that Hilary Clinton was accused of [covering pedophiiles](http://www.collective-evolution.com/2017/09/23/hillary-clinton-covered-up-elite-pedophile-ring-at-state-department-according-to-nbc-news-report/), during the last campaign? Of course, and that is the main point, video evidence will contribute nothing to credibility or deniability of these events **if we know that even that can be completely crafted**.

But can technology itself come to rescue? We discuss two ideas that are often mentioned in this discussion and have existing analogues in today's banknotes. The battle against the proliferation of fake banknotes follows two strategies: (I) make forgery more difficult by introducing elements that are impractical to reproduce and (II) build detectors of fakes notes.



<br>




**I. The crypto way: digital signatures.** In the late twentieth century, advances in computer and photocopy technology made it possible to copy currency easily, without expensive equipment or sophisticated expertise. In response, today banknotes contain hard to copy [features](https://en.wikipedia.org/wiki/Counterfeit_money#Anti-counterfeiting_measures) that such as holograms, multi-coloured bills, embedded devices such as strips, microprinting, watermarks and inks whose colours changes depending on the angle of the light.

Those features serve to make you relative certain about their authenticity since they are impractical to reproduce except by a select few. In other words, you can be fairly certain about their origin being the issuing government.

<p>
<img src="https://www.publicdomainpictures.net/pictures/10000/velka/turkish-banknotes-11282318504l3oc.jpg">
</p>

The problem to address is authentication, which has been a preoccupation of human society [since forever](https://en.wikipedia.org/wiki/History_of_cryptography#Classical_cryptography). How does the commander of a Roman garrison know the orders he just received to abandon his post come from his general and not the enemy general? A message can be encrypted by a [cipher](https://en.wikipedia.org/wiki/Caesar_cipher) and used for authentication: only a person with the right *signing key* could have written it. If only Roman generals possessed the key, the commander can be certain that the message did not come from anyone else.

This concept also exists in the digital world where [cryptographic techniques](https://en.wikipedia.org/wiki/Digital_signature) are widely used to create signatures that everyone can verify, yet only signers with access to a corresponding private signing key can produce. So, how to certify that a digital content is authentic, i.e. it has not been manipulated since it was captured by an electronic device, or even created from scratch?

Similarly to how features are embedded in banknotes, digital signatures can be bound to e.g. a video. The electronic device that captures the content must implement a signing mechanism in hardware. The signature certifies that the video came from the particular device, and no other. In itself this doesn't say anything about the origins of a video, but photographers have for years been interested in cameras that digitally signs photos and videos as part of the shooting process. When implemented securely, anyone can then later verify that e.g. a video was indeed recoded by a camera sensor and that no manipulation has occurred after that point. Importantly, the signature is paired to the original content: editing the video will irremediably invalidate the signature, losing its certificate of authenticity.

There are quite a few drawbacks with digital signatures though. First, they rely on the safety of a private key in hardware. What if an adversary has the access to the device and can temper with it? Second, the same holds about the reliability of the [PKI infrastructure](https://en.wikipedia.org/wiki/Public_key_infrastructure). Third, digital signatures are invalidated by any manipulation of the image, including legitimate editing, such as brightness/contrast or cropping. This is strong limitation in many contexts. Obtaining signing mechanism that are resistant to "benign" editing is an open problem.

In conclusion, there are strong assumptions for this solution to work. Not only we need to trust the infrastructure for the system, we also realistically still need to handle the case where a media is not presented together with its signature: was it acquired by an old device (no signature implemented), or is it a fake? In practical terms, *every* electronic device in circulation capable of recording needs to implement a signing mechanism.

The combined usage of cryptographic hashing and a [public ledger](https://en.wikipedia.org/wiki/Blockchain) for including timestamps to media creation/alteration has been suggested by [this report on AI and national security](https://www.belfercenter.org/sites/default/files/files/publication/AI%20NatSec%20-%20final.pdf). The [location]( https://blog.sldx.com/blockchain-proof-of-location-7af5eb8073c1) at time of recording could also be incorporated into a distributed ledger, providing a verifiable proof that the device was somewhere, and thus a safer mechanisms than a mere hardware signing key. Although, the practicality and effectiveness of those solutions remains unclear.


<br><br>

**II. Machine learning as a solution: building "truth" detectors.**  Back to the banknote parallel, a complementary defence against counterfeit is the use  detectors of fake notes in circulation.

One such tool is an iodine-based ink [detection pen](https://en.wikipedia.org/wiki/Counterfeit_banknote_detection_pen). This is a chemical test on the material that a note is printed on. The special ink is particularly reactive to the paper used with standard printers or photocopiers, while its marks are colourless on genuine banknotes. Notice: this kind of test tells you about authenticity only to some degree of certainty, just like a learning-based system would.

Similarly with digital media, the hypothesis is that forgery leaves traces that are hard to spot by humans, or by humans without the right technical expertise. At the same time, we think that those clues may be detectable by using a machine built for the job.


We can build a model to classify any source of digital media that we believe may be altered. Where does the training data come from? Depending on the particular detection problem you wish to solve, we have essentially the same resources that a forger have: data, cheap computing power, easy-to-access software. Examples for the "fake" class is the output of an algorithm run for to manipulate the media in question -- think of face swapping in videos.

What the defender in this game (who aims to detect fakes) might not know is the particular algorithm used by the attacker (who manipulates the original media). If we believe that most forgers will use what most people can find on the Internet, a defender can do the same and collect training data by running every reasonable tool for forgery.

[Adversarial examples](https://blog.openai.com/adversarial-example-research/) are one last element complicating the scenario. In fact, the attacker's objective is dual: achieving realism for humans (fooling humans) and not being detected by a machine (fooling the machine). [Carlini and Wagner 2017](https://arxiv.org/abs/1705.07263) showed that an attacker can indeed fool several detectors of fake images *if the detector is known during training*. Of course, whether such knowledge is available depends upon many factors in the real world. At the same time, it is also known [[Moosavi-Dezfooli et al. 2017](https://arxiv.org/abs/1610.08401)] that adversarial  images do, to some extent, transfer across different neural networks.

The defender will then need either to invent something new or draw from the latest research; and this applies to the attacker as well... so you see the onset of an arms race. This might make you skeptical that a machine learning based solution is even meaningful in principle, right?

The point is, virtually every computer security problem we face follows the same offence-defence dynamics. A security protocol is implemented and used until somebody breaks it and then a newer, more secure version must be studied and deployed. A great example is the battle between viruses development and anti-virus software firms, and the whole fat industry that monetises on this race. You may even consider *military defence* to work the same way. In those contexts, obviously nobody would bring forward the skeptic argument, against the need of putting in place **the best safety mechanism we can**.

But what is the limit of using ML against ML? If you are familiar with adversarial training and GANs, those are your first objections to using machine learning as a solution.

If you follow this recent Twitter thread, experts opinions seem to polarise in two categories: *undeniable realism will be achieved eventually by generative models* vs. *detection will always be easier than manipulation*:

<blockquote class="twitter-tweet tw-align-center" data-lang="en"><p lang="en" dir="ltr">It is beyond any doubt that over the next few years we will perfect the technology for automatically generating a video of anyone saying anything we type, with the right voice too. What implications do you think this will have? What are the applications? How do we mitigate risks?</p>&mdash; Nando de Freitas (@NandoDF) <a href="https://twitter.com/NandoDF/status/969574632692047872?ref_src=twsrc%5Etfw">March 2, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Just a little more conservative is the new report on [The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf):

> There is no obvious reason why the outputs of these systems could not become indistinguishable from genuine recordings, in the absence of specially designed authentication measures.

While it is hard to say what the limits of generative modelling are in principle in the long run, we envision the attack-defence game dynamic in place in the short-medium term. We must not rule out the potential of implementing learning models for detecting manipulation in digital media, and we advocate it as the first step for a systematic solution. This opinion seems to be shared by experts in the [intelligence community](https://www.belfercenter.org/sites/default/files/files/publication/AI%20NatSec%20-%20final.pdf):

> AI will cause changes in the political security landscape, as the arms race between production and detection of misleading information evolves and states pursue innovative ways of leveraging AI to maintain their rule. It is not clear what the long term implications of such malicious uses of AI will be [...]

In summary, on one hand machine learning as a defence mechanism can be used in principle on any media, regardless on how the source was acquired digitally in the first place. This is a big advantage, at least in the short term, with respect to authentication solutions. On the other hand, by nature of detection systems, this solution will work to some degree of uncertainty and not as a mathematical proof of originality -- unlike a crypto-based solution. Moreover, any learning-based detection system systematically ages and eventually stops working with the improvements of forgeries, unless updated, in an arms race scenario.



<br><br>

**A weekend experiment.** We are not aware of much research work supporting the second solution. [Kashyap et al. 2017](https://arxiv.org/abs/1703.09968) presents a survey on pre-deep learning approaches for detection of image forgery, under several assumptions on the type of manipulation. [D'Avino et al. 2017](http://www.ingentaconnect.com/contentone/ist/ei/2017/00002017/00000007/art00014) proposes a recurrent auto-encoder for anomaly detection in videos. In their experiments, they focus on detecting extraneous objects placed on videos by [green screen acquisition](https://en.wikipedia.org/wiki/Chroma_key). The research area on defences against face spoofing [[Määttä et al. 2011](http://ieeexplore.ieee.org/document/6117510/), [Menotti et al. 2015](http://ieeexplore.ieee.org/abstract/document/7029061/)] is related, although more circumscribed in its goals.

We decided to run a quick experiment over the weekend. Our objective is to verify whether it is feasible to build a statistical model for  distinguishing between real and manipulated images, **when the human eyes and brain would have a hard time to do so**.

To make the task more concrete, we work on Caltech's [Faces '99 dataset](http://www.vision.caltech.edu/html-files/archive.html), which contains 450 frontal face images of about 27 people, with various facial expressions, lighting conditions and background. We set out to swap faces between pictures *of the same person*. This choice, together with having faces all frontal and at the same depth, should make a particularly easy task for the swapping algorithm, in terms of realism of its output. In fact, we aim to show the potential of fake detection in the most difficult scenario we could think of. See some pictures that we will use to test our model:

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/ksDOqEH.jpg)

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/V8XKVuN.jpg)

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/6y6sBH7.jpg)

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/Of6HYn3.jpg)

Can you tell apart real and fake? The solution is below.

The face swap algorithm is from [this open source project](https://github.com/matthewearl/faceswap). We treat it as a black-box and we do not take advantage of any knowledge about it in the experiment.

We split train, validation and test sets by identities, thus we defend against overfitting on particular people faces. Faces swaps are performed within each set. The sizes are respectively of 520, 126, and 170. All those samples are balanced with respect to fake and real images proportion. We additionally create a second larger test set of only fake images, of size 820 -- in fact, we can obtain many more images of face swaps than the original ones (by combination of any pair). On this last test set, we can compute a better estimate of true positive detection, or recall.

The model is a neural network made with off-the-shelf [pytorch vision](https://github.com/pytorch/vision) building blocks. In particular, we use a deep DenseNet, pre-trained on ImageNet, and strong regularization to combat small train set size. The validation set is used to select the model with maximal accuracy. We make an ensemble of 10 of these neural networks, randomly initialized. We achieve 84.7% accuracy on the first test set and measure the same 84.7% for fake recall on the second test set.

To put the numbers into perspective, we ran a few tests on human volunteers. We selected 5 people trained in machine learning or computer vision at various levels (use at work, students, researchers) and 5 people who have none. Those are groups I and II.

The volunteers are presented with 20 pictures, selected at random from the test set. They have unlimited time for classifying an image, but any decision should be taken independently from previously seen ones (this is made clear to the subjects). Notice that we cannot control exactly for independence because humans will start to recognise visual patterns of manipulation by seeing several images sequentially. People are told that this experiment is about recognising face swaps; except this, they are not instructed on what to look for or expect in the image.

We repeat the experiment with two image sizes, a first time with images downsized to 256 x 256 (same as the input to our model) and a second time with the full size of the originals. Results on accuracy are on the table, with an indication of standard deviation:

<br>

| image size        | Group I (ML savvy)| Group II (not ML savvy)|
|:-----------------:|:--------------:|:--------------:|
| 256 x 256         | 57.0 ± 4.0    | 48.0 ± 6.0     |
| 896 x 592 (full)  | 87.0 ± 7.5    | 63.0 ± 17.5     |

<br>

Recall that our model got 84.7% on the low resolution images. That is, it is able to pick up signals of manipulation that the human eye cannot. Although, performance of network and humans seems to match if we provide full size images to the volunteers in group I. Actually, group I with full size images does better than our model *in average*.

There is a marked discrepancy between the two groups. Group II is essentially *tossing a coin* to decide about the low resolution pictures. Moreover, group II cannot beat our model even by analysing full size images and only improve accuracy of 15% in average. This might follow from group I having good priors on what to expect as unrealistic artefacts from a face swap algorithm.

The results show some interesting trends but keep in mind that the sample size of both volunteers and our test set is rather small. Also, there are several factors that condition the numbers. First, we trained our model with hundreds of examples of face swaps, while people didn't see any before the test. Second, the network does classify each image independently, even for the same person in the picture, in contrast with humans that supposedly gets better and better while scanning over the test set twice.

This last point also partially explains the improved performance of the volunteers with full size images: since we use the same set of 20 images, the volunteer can learn to detect manipulation by memorising common patterns and be more confident in spotting them on the second round on the same (but now full size) images. In a way, the higher numbers on the second row depend by both higher resolution and human learning.

At the same time, we did not give the neural network any hint on what to look for. It should be possible to improve performance by detecting face points in the image and augment the input of the model.

*Anyway, take those numbers with a grain of salt*. This is a proof of concept. State of the art techniques should be used to investigate the question more thoroughly. In particular, we are confident that passing those face swaps through a generator of adversarial images would drastically lower the confidence of our detector, unless some adversarial defence is implemented as well. For now, while it is premature to claim any solution to the problem, we got some surprising results: we looked at image forgery by realistic face swapping and it seems **there is some signal for using machine learning when it appears to be little or none for humans**.

What about the images above? Were they real? All four of them are actually fake! Those were hand picked from the test set to showcase particularly hard cases for the human eye. The machine we trained detects 3/4 of them.

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/alex-knight-199368-unsplash.jpg)

A result that actually stands out is the inability of the volunteers to guess better than random, without any prior machine learning knowledge. Did you do better on those four examples?

<br><br>

**Final words.** One day the president of the United States will need to rely on a technology to certify the authenticity of digital evidence from intelligence sources, and take that into consideration before ordering a military strike.

We must start thinking this issue in the same way we look at other fundamental questions, such as climate change. It is inevitable. It will profoundly affect human society. It will re-define the meaning of trust in what we see and hear, anytime  we do not experience it first hand. We need a plan of action.

![]({{ site.url }}/assets/posts/2018-03-17-AI-and-digital-forgery/jesse-orrico-106397-unsplash.jpg)

Wish to get in contact? Reach me at `g.patrini @ uva.nl` or [@giorgiopatrini](https://twitter.com/GiorgioPatrini).


<br><br>

**Acknowledgement.** We are grateful for fantastic feedback on this post from Jorn Peters and Sadaf Gulshad from [DeltaLab](https://ivi.fnwi.uva.nl/uvaboschdeltalab/), Wilko Henecka and Brian Thorne from [N1](https://www.n1analytics.com), Efstratios Gavves, Katy Dynes and Luciano Severgnini. We also thank the ten volunteers for their effort in performing the visual tests.
