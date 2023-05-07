# ML-Visibility-Graphs-for-Alzheimers
Code for analysis, feature selection, and classification of Alzheimer's patient data (and prodromal Alzheimer's) for the paper: [https://www.mdpi.com/2282770](https://www.mdpi.com/2282770).

To get started: You need to install the python3 packages in `requirements.txt`. Everything is self contained in the Jupyter notebook, other than the data. If you have the data, just put the the notebook in the same directory as the folders containing the data (eg "AD_data", "MCI All New Raw Data", etc.) and run it sequentially.


Data availability: The datasets presented in this paper are not readily available because they may contain identifying information and are used with permissions from the Alzheimer’s Disease Research Centers at the University of California, San Diego and the University of California, Davis. Requests to access the datasets should be directed to John Olichney, M.D: [jmolichney@ucdavis.edu]


Please cite our work with the following examples:

```
BIBTEX:


@article{brainsci13050770,
	abstract = {We present a framework for electroencephalography (EEG)-based classification between patients with Alzheimer&rsquo;s Disease (AD) and robust normal elderly (RNE) via a graph theory approach using visibility graphs (VGs). This EEG VG approach is motivated by research that has demonstrated differences between patients with early stage AD and RNE using various features of EEG oscillations or cognitive event-related potentials (ERPs). In the present study, EEG signals recorded during a word repetition experiment were wavelet decomposed into 5 sub-bands (&delta;,&theta;,&alpha;,&beta;,&gamma;). The raw and band-specific signals were then converted to VGs for analysis. Twelve graph features were tested for differences between the AD and RNE groups, and t-tests employed for feature selection. The selected features were then tested for classification using traditional machine learning and deep learning algorithms, achieving a classification accuracy of 100% with linear and non-linear classifiers. We further demonstrated that the same features can be generalized to the classification of mild cognitive impairment (MCI) converters, i.e., prodromal AD, against RNE with a maximum accuracy of 92.5%. Code is released online to allow others to test and reuse this framework.},
	article-number = {770},
	author = {Zhang, Jesse and Xia, Jiangyi and Liu, Xin and Olichney, John},
	date-modified = {2023-05-07 15:12:00 -0700},
	doi = {10.3390/brainsci13050770},
	issn = {2076-3425},
	journal = {Brain Sciences},
	number = {5},
	read = {0},
	title = {Machine Learning on Visibility Graph Features Discriminates the Cognitive Event-Related Potentials of Patients with Early Alzheimer&rsquo;s Disease from Healthy Aging},
	url = {https://www.mdpi.com/2076-3425/13/5/770},
	volume = {13},
	year = {2023},
	bdsk-url-1 = {https://www.mdpi.com/2076-3425/13/5/770},
	bdsk-url-2 = {https://doi.org/10.3390/brainsci13050770}}



DIRECT CITATION:

Zhang, J.; Xia, J.; Liu, X.; Olichney, J. Machine Learning on Visibility Graph Features Discriminates the Cognitive Event-Related Potentials of Patients with Early Alzheimer’s Disease from Healthy Aging. Brain Sci. 2023, 13, 770. https://doi.org/10.3390/brainsci13050770

```
