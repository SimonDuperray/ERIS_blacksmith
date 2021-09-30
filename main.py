from Merger import Merger
from RNN import RNNModel
from Analyzer import Analyzer
from ReportWriter import ReportWriter
from os import listdir
from os.path import join
from functions import average_list
from functions import get_curve_state
from functions import is_good_fitted
from functions import is_under_fitted
from functions import is_over_fitted
import json

# TODO: fitting observations

# ========================
#      ITERATIVE WAY
# ========================

# datasets = [
#     (
#         './datasets/bibtex/5ke/corr_5ke.bibtex',
#         './datasets/bibtex/5ke/incorra_5ke.bibtex',
#         './datasets/bibtex/5ke/incorrb_5ke.bibtex'
#     ), (
#         './datasets/bibtex/10ke/corr_10ke.bibtex',
#         './datasets/bibtex/10ke/incorra_10ke.bibtex',
#         './datasets/bibtex/10ke/incorrb_10ke.bibtex'
#     ), (
#         './datasets/bibtex/15ke/corr_15ke.bibtex',
#         './datasets/bibtex/15ke/incorra_15ke.bibtex',
#         './datasets/bibtex/15ke/incorrb_15ke.bibtex'
#     ), (
#         './datasets/bibtex/20ke/corr_20ke.bibtex',
#         './datasets/bibtex/20ke/incorra_20ke.bibtex',
#         './datasets/bibtex/20ke/incorrb_20ke.bibtex'
#     )
# ]

# small datasets to test algo
datasets = [
    (
        './datasets/bibtex/10ke/corr_10ke.bibtex',
        './datasets/bibtex/10ke/incorra_10ke.bibtex',
        './datasets/bibtex/10ke/incorrb_10ke.bibtex'
    )
]

# gru_neurons = [32, 64, 128, 256, 512]
gru_neurons = [64, 128, 256, 512]

epochs = [10, 15]

for dataset in datasets:
    # create dataset
    merger = Merger()
    merger.run(
        path_1=dataset[0],
        path_2=dataset[1],
        path_3=dataset[2]
    )

    # instanciate new ReportWriter
    report_writer = ReportWriter()

    # report title
    report_title = "Experimental Report for dt_lgth:" + str(merger.dflgth)
    filename = "experimental_report_" + str(merger.dflgth) + ".md"

    # write the first page of the report
    report_writer.write_first_page(
        filename=filename,
        title=report_title,
        iter_data={"gru_neurons": gru_neurons, "epochs": epochs},
        params={
            "batch_size ": 300,
            "validation_split ": 0.2,
            "dataset_path ": "./datasets/",
            "learning_rate ": 1e-3,
            "dropout_value ": 0.5,
            "dense_neurons ": 128,
            "loss_function ": "sparse_categorical_crossentropy",
            "dataset_length ": str(merger.dflgth),
            "percentage True Negatives ": str(merger.percentage),
            "Percentage title ": str((16.58+16.66)/2),
            "percentage other tags ": str(0)
        }
    )

    # train models
    for gru_neuron in gru_neurons:
        for epoch in epochs:
            # instanciate new model
            rnn_model = RNNModel()

            # then train it
            rnn_model.run(
                nb_epochs=epoch,
                batch_size=300,
                validation_split=0.2,
                dataset_path=merger.get_dataset_location(),
                gru_neurons=gru_neuron,
                learning_rate=1e-3,
                dropout_value=0.5,
                dense_neurons=256,
                loss_function=None,
                gru_neuron=gru_neuron
            )

            # instanciate new Analyzer()
            analyzer = Analyzer(epochs=epoch)  # OR EPOCH ?

            # get current folder's name
            directories = listdir("./histories/" + str(merger.dflgth))
            for directory in directories:
                if int(str(directory[2])) == epoch:
                    dir = directory
                    files = listdir(join("./histories/" + str(merger.dflgth) + "/", directory))
                else:
                    try:
                        if int(str(directory[2]) + str(directory[3])) == epoch:
                            dir = directory
                            files = listdir(join("./histories/" + str(merger.dflgth) + "/", directory))
                        else:
                            pass
                    except ValueError as e:
                        print(f"Error: {e}")

                file = files[0] if files[0][-4:] == "json" else files[1]

            # get path for acc and loss
            path_acc = "./analysis/compar_epochs/" + str(merger.dflgth) + "/acc_valacc_" + str(epoch) + "_" + str(
                gru_neuron) + ".png"
            path_loss = "./analysis/compar_epochs/" + str(merger.dflgth) + "/loss_valloss_" + str(epoch) + "_" + str(
                gru_neuron) + ".png"

            # analyze metrics and save model and json files
            analyzer.analyze_for_report(
                dir=dir,
                file=file,
                dtlgth=merger.dflgth,
                path_acc=path_acc,
                path_loss=path_loss
            )

            # get data from file
            opened_file = "./histories/" + str(merger.dflgth) + "/" + dir + "/" + file
            print(f"\n==========Opened file: {opened_file}\n==========")
            with open("./histories/" + str(merger.dflgth) + "/" + dir + "/" + file, 'r') as outfile:
                data = json.load(outfile)
                training_loss = data['loss']
                validation_loss = data['val_loss']

            print("==========\n")
            print(data)
            print("\n==========")

            print("\n==========\n")
            print(data['accuracy'][-1], data['accuracy'][-2])
            print("\n==========\n")
            print(data['val_accuracy'][-1], data['val_accuracy'][-2])
            print("\n==========\n")
            print(data['loss'][-1], data['loss'][-2])
            print("\n==========\n")
            print(data['val_loss'][-1], data['val_loss'][-2])
            print("\n==========\n")

            curves_data = {}
            curves_data['accuracy'] = {
                "tr_min": min(data['accuracy']),
                "tr_max": max(data['accuracy']),
                "tr_mean": average_list(data['accuracy']),
                "tr_state": get_curve_state(data['accuracy'][-1], data['accuracy'][-2]),
                "val_min": min(data['val_accuracy']),
                "val_max": max(data['val_accuracy']),
                "val_mean": average_list(data['val_accuracy']),
                "val_state": get_curve_state(data['val_accuracy'][-1], data['val_accuracy'][-2])
            }
            curves_data['loss'] = {
                "tr_min": min(data['loss']),
                "tr_max": max(data['loss']),
                "tr_mean": average_list(data['loss']),
                "tr_state": get_curve_state(data['loss'][-1], data['loss'][-2]),
                "val_min": min(data['val_loss']),
                "val_max": max(data['val_loss']),
                "val_mean": average_list(data['val_loss']),
                "val_state": get_curve_state(data['val_loss'][-1], data['val_loss'][-2])
            }

            fitting_observations = {
                "under": is_under_fitted(training_loss, validation_loss),
                "over": is_over_fitted(training_loss, validation_loss),
                "good": is_good_fitted(training_loss, validation_loss) 
            }

            predictions = {
                'to_predict': data['init_seq'],
                'expected': data['to_predict'],
                'predicted': data['prediction']
            }

            print("\n\n\n======\n" + str(rnn_model.get_to_predict().split()) + "\n======\n\n\n")
            print("\n\n\n======\n" + str(rnn_model.get_predicted().split()) + "\n======\n\n\n")

            # determine prediction percentage confidence
            # TODO: create function in Analyzer (later)
            predicted = predictions['predicted'].split()
            expected = predictions['expected'].split()

            if len(predicted)<len(expected):
                for i in range(len(expected)-len(predicted)):
                    predicted.append('...')
            eq = 0
            for exp, pre in zip(expected, predicted):
                if exp == pre:
                    eq += 1
            
            percentage_confidence = int(round(float(eq*100/len(expected)), 0))

            bilan = {
                "gap_acc": max(data['accuracy']),
                "gap_loss": min(data['loss']),
                "fit_status": "underfitted",
                "pred_percent": percentage_confidence
            }

            # append new page to report
            report_writer.write_unique_epoch_page(
                gru_neuron=gru_neuron,
                epoch=epoch,
                acc_valacc_absolute_path="." + path_acc,
                loss_valloss_absolute_path="." + path_loss,
                curves_data=curves_data,
                fitting_observations=fitting_observations,
                predictions=predictions,
                bilan=bilan
            )

        report_writer.write_last_page(
            dtlgth=str(merger.dflgth),
            gru_neuron=gru_neuron
        )
