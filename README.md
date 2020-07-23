Supplementary Material used in "Memory networks for consumer protection:unfairness exposed"

# Enviroment settings

A requirement.txt file is located at project level to allow fast python enviroment setup (hope that the package listing is exhaustive). All tests were run with tensorflow-gpu 2.0 and CUDA 10.0.

# Cross-validation Training

Running a training test is quite simple:

* Open runnables/quick_test_setup.py and setup script variables as you please [lines 17-42] (check the script documentation inside the file).
* Run runnables/quick_test_setup.py. This script setups all test related configuration files except for configs/callbacks.json and configs/distributed_model_config.json. The former contains early stopping callback settings, whereas the latter stores model hyper-parameters.
* Once you are satisfied with all configuration files, run runnables/test_cross_validation_v2.py.
* If test settings allow model saving, all test results and associated meta-data will be saved within cv_test folder.


# Visualizing training plots

If cross-validation routine settings allow model saving, a specific training callback is enabled as well. At the end of the training procedure, it is possible to visualize all training curves (i.e. train and validation losses and metrics).

* Open runnables/visualize_training_behaviours.py and setup script variables as you please [lines 36-37] (check the script documentation inside the file). In particular, if the model is a memory network with strong supervision, you can uncomment lines [40-41, 51-52] in order to visualize strong supervision losses as well.
* Run runnables/visualize_training_behaviours.py


# Computing cross-validation metrics

Once a training routine has successfully ended, it is possible to quickly print overall metric values:

* Open runnables/compute_cv_stats.py and setup script variables as you please [lines 84-88] (check the script documentation inside the file).
* Run runnables/compute_cv_stats.py and the following stats should be printed: validation metrics, test metrics, ensemble metrics (if more than 1 repetitions).


# Gathering memory attention

In order to collect memory stats, it is necessary to retrieve memory attention distribution beforehand.

* Open runnables/gather_attention_distributions_v2.py and setup script variables as you please [lines 46-53] (check the script documentation inside the file). In particular, it is sufficient to specify 'model_type' and 'test_name' variables.
* Run runnables/gather_attention_distributions_v2.py and specified attention weights should be stored within model test folder.


# Retrieving memory stats

Once model memory attention has been collected and stored, it is possible to compute all described memory stats and metrics

* Open runnables/visualize_voting_coverage.py and setup script variables as you please [lines 109-115] (check the script documentation inside the file).
* Run runnables/visualize_voting_coverage.py and memory stats should be printed for each of the following models: each per fold best model up to K and union and intersection of aforementioned models.


# Retrieving dataset text stats

Checking dataset textual statistics concerning sentence length is straightforward.

* Open runnables/compute_data_stats.py and modify the filter variable at line 15 (choose one of the following categories: A, CH, CR, TER, LTD).
* Run runnables/compute_data_stats.py


# Enabling strong supervision

Strong supervision come as an additional set of model hyper-parameters. In particular, please check the **partial_supervision_info** setting within configs/distributed_model_config.json -> 'experimental_basic_memn2n_v2'.

```
{
	[...]
	"partial_supervision_info": {
		"value": {
			"flag": true|false,					# enable/disables strong supervision
			"coefficient": 1.0,					# regularization coefficient trading-off strong supervision contribution (0. -> no effect)
			"margin": 0.5,						# this is the \gamma value in Eq (2)
			"mask_by_attention": false 			# currently fixed at false (not used)
		}
	}
}

```


# Inspecting legal rationales (i.e the KB)

You can check legal rationales for each unfairness category in local_database/KB.


# Inspecting ToS 100 dataset

You can check Terms of Service (ToS) dataset, comprised of 100 documents, in local_database/ToS_100. The data is in .csv format and contains the following columns:

* A, CH, CR, J, LAW, LTD, PINC, TER, USE: a column for each identified unfairness category. **Note: we only have legal rationales for 5 of these unfairness categories: A, CH, CR, LTD, TER. Future work might involve remaining unfairness categories.** 
* document: name of the terms of service document
* document_ID: unique identified of the document (integer)
* label: general unfairness label -> 1 if at least one unfairness category label is set to 1, 0 otherwise
* text: sentence text as processed by Stanford CoreNLP
* A|CH|CR|LTD|TER_targets: each sentence labelled as unfair and belonging to one the aforementioned unfairness categories is further annotated with the ground truth set of legal rationales. In particular, the set is encoded as the list of positional indexes w.r.t. KB .txt files (check local_database/KB/\*.txt files). For example [0, 5, 7] set for the category A is equivalent to saying *the first, the sixth and the eighth sentences as listed in A.txt*.
