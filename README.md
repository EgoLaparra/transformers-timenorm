# Transformers timenorm

The first time you run the code, the pre-trained model, `clulab/roberta-timex-semeval`, will be automatically downloaded in your computer. If you want to produce some predictions with this model, you need to pass as arguments the directory containing the input text and the target directory where the predictions will be stored. For example, to process the _AQUAINT_ subset from _TempEval-2013_, just run:

    python run_time.py -p /paht/to/anafora-annotation/TempEval-2013/AQUAINT -o /path/to/output/AQUAINT [--no_cuda]

Recall that the `anafora-annotation` folder includes both raw text and Anafora annotation files, but it could contain only the former since the latter are not needed to make predictions. This will be the case during the evaluation phase.

You can continue training the pre-trained model from its current checkpoint if you dispose of additional annotated data by running the following command:

    python run_time.py -t /path/to/train-data/ -s /path/to/save-model/ [--no_cuda]

The `train-data` directory must follow a similar structure to the _AQUAINT_ or _TimeBank_ folders and include, for each document, both the raw text file (with no extension) and the Anafora annotation file (with `.xml` extension). After running the training, the `save-model` will contain two sub-folders, `logs`, with a set of log files that can be visualized with _TensorBoard_, and `results`, that contains all the checkpoints saved during the training and three files (`pytorch_model.bin`, `training_args.bin` and `config.json`) with the configuration and weights of the final model.

To use this new version of the model for predictions, you can run:

    python run_time.py -p /paht/to/text/TempEval-2013/AQUAINT -o /path/to/output/AQUAINT -m /path/to/save-model/results/ [--no_cuda]

If you still want to continue the training from this point, just run:

    python run_time.py -t /path/to/train-data/ -s /path/to/save-model-2/ -m /path/to/save-model/results/ [--no_cuda]

Use the `--no_cuda` option if you are going to run the commands above in the gpu.  The `/path/to/save-model/results/` value for the `-m` option can be also replaced by a model stored in the HuggingFace hub. E.g. `-m clulab/fake-timex-model`.

Run `python run_time.py -h` to explore additional options and arguments you can play with, like the hyperpameters of the model. 
