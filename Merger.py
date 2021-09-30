from Transformation import CustomTransformation
import random
import csv
import pandas as pd
import collections


class Merger:

    def __init__(self):
        self.input_ = None
        self.output_ = None
        self.percentage = 0
        self.fieldnames = ['in', 'out']
        self.rows = []
        self.dflgth = 0
        self.dataset_location = ""
        self.percentage_names_only = None
        self.other_tags = None

    @staticmethod
    def contains_letter(word):
        return word.lower().islower()

    def preprocessing(self, li):
        striped = [i.strip() for i in li]
        for i in range(len(striped)):
            if not self.contains_letter(word=li[i]):
                del striped[i]
        return [item for item in striped if item[:8] == '<authors']

    @staticmethod
    def replace_quotes(li):
        for i in range(len(li)):
            if '"' in li[i]:
                li[i] = li[i].replace('"', "'")
        return li

    @staticmethod
    def add_spaces_str(word):
        if '"' in word:
            word = word.replace('"', ' " ')
        if "'" in word:
            word = word.replace("'", " ' ")
        return word

    @staticmethod
    def add_spaces_list(li):
        for i in range(len(li)):
            if "'" in li[i]:
                li[i] = li[i].replace("'", " ' ")
            if '"' in li[i]:
                li[i] = li[i].replace('"', ' " ')
        return li

    @staticmethod
    def read_bibtex_file(filename):
        with open(filename, "r") as file:
            return [line.strip('\n') for line in file]

    def get_percent(self):
        return self.percentage

    def get_dataset_location(self):
        return self.dataset_location

    def get_i_o_length(self):
        return len(self.input_) if len(self.input_) == len(self.output_) else min(len(self.input_), len(self.output_))

    def run(self, path_1, path_2, path_3):
        # open files
        correct_in = self.read_bibtex_file(filename=path_1)
        incorrect_a_in = self.read_bibtex_file(filename=path_2)
        incorrect_b_in = self.read_bibtex_file(filename=path_3)

        # preprocess data
        correct_in = self.preprocessing(correct_in)
        incorrect_a_in = self.preprocessing(incorrect_a_in)
        incorrect_b_in = self.preprocessing(incorrect_b_in)

        # replace " by '
        correct_in = self.replace_quotes(li=correct_in)
        incorrect_a_in = self.replace_quotes(li=incorrect_a_in)
        incorrect_b_in = self.replace_quotes(li=incorrect_b_in)

        # apply custom transformation
        transformer = CustomTransformation()
        correct_out = transformer.transform(correct_in)
        incorrect_a_out = transformer.transform(incorrect_a_in)
        incorrect_b_out = transformer.transform(incorrect_b_in)

        # delete shift
        shift = abs(len(incorrect_a_out) - len(incorrect_b_out))
        biggest = [incorrect_a_in, incorrect_a_out] \
            if len(incorrect_a_out) > len(incorrect_b_out) \
            else [incorrect_b_in, incorrect_b_out]

        for i in range(shift):
            for li in biggest:
                del li[-1]

        # create final lists
        self.input_ = correct_in + incorrect_a_in[:len(incorrect_a_in)]
        self.output_ = correct_out + incorrect_b_out[:len(incorrect_b_out)]

        if len(self.input_) == len(self.output_):
            self.dflgth = len(self.input_)
        else:
            self.dflgth = len(min(self.input_, self.output_))

        # shuffle lists
        random.Random(4).shuffle(self.input_)
        random.Random(4).shuffle(self.output_)

        incorr, tt = 0, 0
        r = []
        if len(self.input_) == len(self.output_):
            for i in range(len(self.input_)):
                if self.input_[i][17:-3] != self.output_[i][8:-1]:
                    incorr += 1
                tt += 1
                r.append(
                    {
                        "in": self.add_spaces_str(word=self.input_[i]),
                        "out": self.add_spaces_str(word=self.output_[i])
                    }
                )

        self.percentage = round(100 * incorr / tt, 1)
        self.rows = r

        print("=======================================================================================================")
        print("[Merging running...]")
        print(f"Preprocessing successfully operated !\n>>> "
              f"Percentage of true negatives: {self.percentage}%\n[next step]")

        self.input_ = self.add_spaces_list(li=self.input_)
        self.output_ = self.add_spaces_list(li=self.output_)

        # save file
        filename = 'dataset'+str(self.dflgth)+'.csv'
        path = "./datasets/"
        self.dataset_location = path+filename
        with open(path + filename, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        # log message
        print(f"{filename} saved in {path} !")
