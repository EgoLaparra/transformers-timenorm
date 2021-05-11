import argparse
from transformers import (
    T5Config,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from utils_time import *


class DatasetEmptyError(Exception):

    def __init__(self, dir_name):
        self.message = dir_name

    def __str__(self):
        return "Dataset empty. Check the structure of your %s directory." % self.message


class Model:

    def __init__(self, args):
        self.config = T5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        if args.vocab_file is not None and args.merges_file is not None:
            self.tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_name_or_path, config=self.config,
                                                             cache_dir=args.cache_dir,
                                                             vocab_file=args.vocab_file,
                                                             merges_file=args.merges_file,
                                                             model_max_length=args.max_seq_length,
                                                             use_fast=True)
        else:
            self.tokenizer = T5TokenizerFast.from_pretrained("t5-small", config=self.config,
                                                             cache_dir=args.cache_dir,
                                                             model_max_length=args.max_seq_length,
                                                             use_fast=True)
        bio_mode = not args.io_mode
        self.config.label_as_token = args.label_as_token
        if args.label_as_token:
            self.config.label2token = read_t5_labels("resources/labels.txt", self.tokenizer, bio_mode=bio_mode)
            self.config.token2label = dict((token, label) for label, token in self.config.label2token.items())
        else:
            labels = read_labels("resources/labels.txt", bio_mode=bio_mode)
            self.config.id2label = dict((idx, label) for idx, label in enumerate(labels))
            self.config.label2id = dict((label, idx) for idx, label in enumerate(labels))
        self.config.bio_mode = bio_mode
        self.config.pad_labels = args.ignore_index
        self.config.label_pad_id = torch.nn.CrossEntropyLoss().ignore_index
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=self.config,
                                                                cache_dir=args.cache_dir)
        output_path = "./runs"
        if args.save_dir is not None:
            output_path = os.path.join(args.save_dir, "results")
        elif args.output_dir is not None:
            output_path = args.output_dir
        logs_path = os.path.join(args.save_dir, "logs") if args.save_dir is not None else None
        self.args = Seq2SeqTrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=args.overwrite_output_dir,
            logging_dir=logs_path,
            no_cuda=args.no_cuda,
            seed=args.seed,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            adam_epsilon=args.adam_epsilon,
            save_steps=args.save_steps,
            evaluation_strategy="steps" if args.validate else "no",
            eval_steps=args.eval_steps,
            max_grad_norm=args.max_grad_norm,
            local_rank=args.local_rank,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level
        )

    @staticmethod
    def train_data_collator(features):
        batch = dict()
        batch["input_ids"] = torch.stack([f.input_ids for f in features])
        batch["attention_mask"] = torch.stack([f.attention_mask for f in features])
        batch["labels"] = torch.stack([f.label for f in features])
        return batch

    @staticmethod
    def test_data_collator(features):
        batch = dict()
        batch["input_ids"] = torch.stack([f.input_ids for f in features])
        batch["attention_mask"] = torch.stack([f.attention_mask for f in features])
        batch["decoder_input_ids"] = torch.stack([f.decoder_input_ids for f in features])
        return batch

    @staticmethod
    def compute_metrics(dataset):
        def anafora_evaluation(prediction):
            scores = score_predictions(model, dataset, prediction.predictions)
            return {
                "precision": scores.precision(),
                "recall": scores.recall(),
                "f1": scores.f1()
            }
        return anafora_evaluation

    def train(self, train_dataset, eval_dataset=None):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics(eval_dataset),
            data_collator=self.train_data_collator
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)

    def evaluate(self, dataset):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.args,
            eval_dataset=dataset,
            compute_metrics=self.compute_metrics(dataset),
            data_collator=self.train_data_collator
        )
        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)
        trainer.evaluate()

    def predict(self, dataset):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.args,
            data_collator=self.test_data_collator
        )
        prediction, _, _ = trainer.predict(dataset)
        return prediction

    def generate(self, dataset):
        input_ids = torch.stack([f.input_ids for f in dataset.features])
        prediction = self.model.generate(input_ids=input_ids, max_length=500)
        return prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""%(prog)s runs SEMEVAL-2021 temporal baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", metavar="DIR", dest="train_dir",
                        help="The root of the training set directory tree containing raw text and of Anafora XML.")
    parser.add_argument("-v", "--valid", metavar="DIR", dest="valid_dir",
                        help="The root of the validation set directory tree containing raw text and of Anafora XML.")
    parser.add_argument("-p", "--predict", metavar="DIR", dest="predict_dir",
                        help="The root of the directory tree containing raw text for prediction.")
    parser.add_argument("-o", "--output", metavar="DIR", dest="output_dir",
                        help="The directory to store the prediction in Anafora XML.")
    parser.add_argument("-s", "--save", metavar="DIR", dest="save_dir",
                        help="The directory to save the model and the log files.")
    parser.add_argument("-m", "--model", metavar="NAME|PATH", dest="model_name_or_path",
                        default="clulab/roberta-timex-semeval", help="Name or path to a trained model.")
    parser.add_argument("-k", "--tokenizer", metavar="NAME|PATH", dest="tokenizer_name_or_path",
                        default=None, help="Name or path to a tokenizer. If None use the model name.")
    parser.add_argument("-c", "--cache", metavar="NAME|PATH", dest="cache_dir",
                        help="Directory in which a downloaded pre-trained model  should be cached.")
    parser.add_argument("-b", "--vocab", metavar="NAME|PATH", dest="vocab_file", default=None,
                        help="Name or path to a vocabulary file.")
    parser.add_argument("-r", "--merges", metavar="NAME|PATH", dest="merges_file", default=None,
                        help="Name or path to a merges file.")
    parser.add_argument("-i", "--io", dest="io_mode", action="store_true",
                        help="Use IO labelling instead of BIO.")
    parser.add_argument("-l", "--as_token", dest="label_as_token", action="store_true",
                        help="Translate label to T5 starting tokens.")
    parser.add_argument("-g", "--ignore", dest="ignore_index", action="store_true",
                        help="Use ignore index, padded labels do not contribute to the loss.")

    hyper = parser.add_argument_group("hyper_parameters")
    hyper.add_argument("--no_cuda", action="store_true", help=" ")
    hyper.add_argument("--max_seq_length", default=512, type=int, help=" ")
    hyper.add_argument("--train_batch_size", default=2, type=int, help=" ")
    hyper.add_argument("--eval_batch_size", default=2, type=int, help=" ")
    hyper.add_argument("--learning_rate", default=5e-5, type=float, help=" ")
    hyper.add_argument("--num_train_epochs", default=3.0, type=float, help=" ")
    hyper.add_argument("--warmup_steps", default=500, type=int, help=" ")
    hyper.add_argument("--weight_decay", default=0.01, type=float, help=" ")
    hyper.add_argument("--adam_epsilon", default=1e-8, type=float, help=" ")
    hyper.add_argument("--save_steps", default=500, type=int, help=" ")
    hyper.add_argument("--eval_steps", default=500, type=int, help=" ")
    hyper.add_argument("--validate", action="store_true", help=" ")
    hyper.add_argument("--overwrite_output_dir", action="store_true", help=" ")
    hyper.add_argument("--max_grad_norm", default=1.0, type=float, help=" ")
    hyper.add_argument("--local_rank", default=-1, type=int, help=" ")
    hyper.add_argument("--seed", default=42, type=int, help=" ")
    hyper.add_argument("--gradient_accumulation_steps", default=1, type=int, help=" ")
    hyper.add_argument("--fp16", action="store_true", help=" ")
    hyper.add_argument("--fp16_opt_level", default="O1", type=str, help=" ")

    args = parser.parse_args()
    train_path = args.train_dir
    valid_path = args.valid_dir
    predict_path = args.predict_dir
    output_path = args.output_dir
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    model = Model(args)
    nlp = init_nlp_pipeline()

    if train_path is not None and valid_path is not None:
        train_dataset = create_datasets(model, nlp, train_path, train=True)
        if len(train_dataset) == 0:
            raise DatasetEmptyError("train")
        valid_dataset = create_datasets(model, nlp, valid_path, valid=True)
        if len(valid_dataset) == 0:
            raise DatasetEmptyError("validation")
        model.train(train_dataset, eval_dataset=valid_dataset)
        if output_path is not None:
            model.predict(valid_dataset)
            write_predictions(model, valid_dataset, output_path)

    elif train_path is not None:
        train_dataset = create_datasets(model, nlp, train_path, train=True)
        if len(train_dataset) == 0:
            raise DatasetEmptyError("train")
        model.train(train_dataset)

    elif valid_path is not None:
        valid_dataset = create_datasets(model, nlp, valid_path, valid=True)
        if len(valid_dataset) == 0:
            raise DatasetEmptyError("validation")
        model.evaluate(valid_dataset)

    elif predict_path is not None and output_path is not None:
        test_dataset = create_datasets(model, nlp, predict_path)
        if len(test_dataset) == 0:
            raise DatasetEmptyError("text")
        prediction = model.generate(test_dataset)
        write_predictions(model, test_dataset, prediction, output_path)

    else:
        print("Invalid option combination.\n")
        parser.print_help()
