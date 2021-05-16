Encoder-Decoder: Sparsity & Sentence Structure
=====================================================
Requirements
--------------------------------------
- python 3.7
- torch 1.2.0
- transformers (HuggingFace) 2.11.0

Overview
--------------------------------------
1. train_[...].py = training scripts
2. decode.py = running decoding (inference)
3. validate_[...].py = validation for KL & integrated training
4. models/ = scripts for model, e.g. modeling
5. conf/ = configuration files for training
6. lobart/ = sub dir for all the work using LoBART

Pipeline (before training starts)
--------------------------------------
- **Must-do**: Copy ```models/modeling_bart_efficient_decoder.py``` to where you have ```transformers``` library installed.

		cp models/modeling_bart_efficient_decoder.py PATH_TO_TRANSFORMERS/.	
- Data: We use CNNDM & XSum from HuggingFace's datasets library.
- Steps: Train -> Valid -> Decode

Training
--------------------------------------
**Forcing Sparisity**:

    python train_sparse.py conf/sparse_CNNDM.txt

**KL-only**:

    python train_KL.py conf/KL_CNNDM.txt

**Integrated Training**:

    python train_integrated.py conf/integrated_CNNDM.txt

Configurations are set in the ```config.txt``` file:

- **bart_weights** - pre-trained BART weights, e.g. facebook/bart-large-cnn
- **bart_tokenizer** - pre-trained tokenizer, e.g. facebook/bart-large
- **model_name** - model name to be saved
- **save_dir** - directory to save checkpoints
- **task** - CNNDM, XSUM
- **optimizer** - optimzer (currently only adam supported)
- **max\_target\_len** - maximum target length
- **lr0**  - lr0
- **warmup** - warmup
- **batch_size** - batch_size
- **gradient_accum** - gradient_accum
- **valid_step** - save a checkpoint every ...
- **total_step** - maximum training steps
- **early_stop** - stop training if validaation loss stops improving for ... times
- **random_seed** - random_seed
- **use_gpu** - True | False
- **num_heads** - 16 (for BART)
- **num_layers** - 12 (for BART)
- **eos_id** - sentence boundary token id (4 for CNNDM, XSUM, Podcast, and 479 for arXiv)
- **load\_model\_path** = to load a model

Sparsity specific config:

- **gamma** - 0.1 (multitask)

KL-only & Integrated training specific  config:

- **temperature** - 0.5
- **eps** - 1e-8

Integrated training specific  config:

- **lambda1** - 0.2 (multitask)
- **r_train** - no. sentences retained at training time
- **training_ref** - exact,approx,mix

Decoding (Inference)
--------------------------------------
**sparisity/KL-only/integrated-intraing**:

	python decode.py \
		--decode_type [ideal|model_based|model_free|random]
		--load model_checkpoint
		--decode_dir output_dir
		--dataset [CNNDM|XSUM]
		--start_id int
		--end_id int
		--r_inference int
		[--num_beams NUM_BEAMS]
		[--max_length MAX_LENGTH]
		[--min_length MIN_LENGTH]
		[--no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE]
		[--length_penalty LENGTH_PENALTY]
		[--random_order [RANDOM_ORDER]]
		[--use_gpu [True|False]]

**baseline all attention**:

	python decode_baseline_allattn.py \
		--load model_checkpoint
		--decode_dir output_dir
		--dataset [CNNDM|XSUM]
		--start_id int
		--end_id int
		[--num_beams NUM_BEAMS]
		[--max_length MAX_LENGTH]
		[--min_length MIN_LENGTH]
		[--no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE]
		[--length_penalty LENGTH_PENALTY]
		[--random_order [RANDOM_ORDER]]
		[--use_gpu [True|False]]

Validation
--------------------------------------
**Sparsity Training**: validation function is included in the training script

**KL-only/integrated-training** - Do this separately to finish training faster!! e.g.:

	python validate_integrated.py \
		--load model_checkpoint
		--config_path path_to_train_config
		--cache_dir dir_to_write_out_valid_loss
		--start_id int
		--end_id int
		[--random_order [RANDOM_ORDER]]
		[--use_gpu [True|False]]

Note that validation loss for each validation instance will be written to a text file e.g. temp/0_vloss.txt. Just write a script to compute the average of the entire validation set. Do the same for ```validation_KL.py```


LoBART Experiments
-----------------------------------------
```lobart_work/``` has a similar structure to this main repository. Training is done in the same fashion, but decoding scripts currently need manual setting in the script (see corresponding variable names).
