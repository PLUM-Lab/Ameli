import logging
import os
import csv
from typing import List
from sentence_transformers import  InputExample
import numpy as np
import pickle



logger = logging.getLogger(__name__)

class CESoftmaxAccuracyEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.

    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int],guid_list: List[str], name: str='', write_csv: bool = True,
                 show_progress_bar: bool = False,batch_size=32,is_save_score=False):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name

        self.csv_file = "CESoftmaxAccuracyEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy"]
        self.write_csv = write_csv
        self.batch_size=batch_size
        self.show_progress_bar=show_progress_bar
        self.is_save_score=is_save_score
        self.guid_list=guid_list
        self.score_file="CESoftmaxAccuracyEvaluator"+ ("_" + name if name else '') + "_score.pickle"

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []
        guid_list=[]
        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
            guid_list.append(example.guid)
        return cls(sentence_pairs, labels,guid_list, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CESoftmaxAccuracyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar,batch_size=self.batch_size)
        pred_labels = np.argmax(pred_scores, axis=1)
        
        assert len(pred_labels) == len(self.labels)

        acc = np.sum(pred_labels == self.labels) / len(self.labels)

        logger.info("Accuracy: {:.2f}".format(acc*100))

        if output_path is not None :
            if self.write_csv:
                csv_path = os.path.join(output_path, self.csv_file)
                output_file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(self.csv_headers)

                    writer.writerow([epoch, steps, acc])
            if self.is_save_score:
                score_dict={"pred_scores":pred_scores,"pred_labels":pred_labels,"gold_label":self.labels,
                            "guid":self.guid_list}
                score_path=os.path.join(output_path, self.score_file)
                with open(score_path, 'wb') as handle:
                    pickle.dump(score_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return acc
