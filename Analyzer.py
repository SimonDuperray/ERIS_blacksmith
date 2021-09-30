from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import json
import collections

class Analyzer:

    def __init__(self, epochs):
        self.epochs_list = epochs

    @staticmethod
    def analyze_for_report(dir, file, dtlgth, path_acc, path_loss):
        obj = {}
        with open("./histories/"+str(dtlgth)+"/"+dir+"/"+file, 'r') as outfile:
            data = json.load(outfile)
            obj['epochs'] = [i+1 for i in range(data['epochs'])]
            obj['accuracy'] = data['accuracy']
            obj['val_accuracy'] = data['val_accuracy']
            obj['loss'] = data['loss']
            obj['val_loss'] = data['val_loss']

        # acc / val_acc
        plt.plot(obj['epochs'], obj['accuracy'])
        plt.plot(obj['epochs'], obj['val_accuracy'], linestyle='--')
        plt.title("Training and Validation Accuracy")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.scatter(obj['epochs'][-1], obj['accuracy'][-1], color='red')
        plt.scatter(obj['epochs'][-1], obj['val_accuracy'][-1], color='red')
        plt.plot([obj['epochs'][-1], obj['epochs'][-1]], [obj['accuracy'][-1], obj['val_accuracy'][-1]], color='red')
        plt.legend(['train_acc', 'val_acc'])
        plt.grid()
        delta = abs(obj['accuracy'][-1]-obj['val_accuracy'][-1])
        txt = r'$\Delta='+str(delta)+"$"
        text_x_pos = 2 + float(obj['epochs'][-1]) * 7 / 80
        plt.text(
            min(obj['epochs']),
            (103*min(obj['accuracy']))/100,
            txt,
            fontsize=8
        )
        plt.savefig(path_acc)
        plt.show()
        plt.close()

        # loss / val_loss
        plt.plot(obj['epochs'], obj['loss'])
        plt.plot(obj['epochs'], obj['val_loss'], linestyle='--')
        plt.title("Training and Validation Loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.scatter(obj['epochs'][-1], obj['loss'][-1], color='red')
        plt.scatter(obj['epochs'][-1], obj['val_loss'][-1], color='red')
        plt.plot([obj['epochs'][-1], obj['epochs'][-1]], [obj['loss'][-1], obj['val_loss'][-1]], color='red')
        plt.legend(['train_loss', 'val_loss'])
        plt.grid()
        delta = abs(obj['loss'][-1]-obj['val_loss'][-1])
        txt = f'$\Delta='+str(delta)+"$"
        text_x_pos = 2 + float(obj['epochs'][-1]) * 7 / 80
        plt.text(
            min(obj['epochs']),
            (103*min(obj['loss']))/100,
            txt,
            fontsize=8
        )
        plt.savefig(path_loss)
        plt.show()
        plt.close()

    def analyze_epochs(self, subfolder):
        list_of_dirs = listdir("./histories/"+subfolder)
        results = []
        for dirr in list_of_dirs:
            files = listdir(join('./histories/'+subfolder, dirr))
            if files[0][-4:] == "json":
                file_to_open = files[0]
            else:
                file_to_open = files[1]
            with open("./histories/"+subfolder+"/" + dirr + "/" + file_to_open, 'r') as file:
                data = json.load(file)
                results.append(
                    {
                        "epochs": [i + 1 for i in range(data['epochs'])],
                        "accuracy": data['accuracy'],
                        "val_accuracy": data['val_accuracy'],
                        "loss": data['loss'],
                        "val_loss": data['val_loss']
                    }
                )

        epochs_list = []
        for item in results:
            epochs_list.append(len(item['epochs']))
        max_epochs = max(epochs_list)
        epochss = [i + 1 for i in range(max_epochs)]

        # accuracy
        for item in results:
            plt.plot(epochss, item['accuracy'] + [None for _ in range(max_epochs - len(item['accuracy']))])
        plt.legend([str(i) + " epochs" for i in self.epochs_list])
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Evolution of accuracy across epochs')
        plt.savefig("./analysis/compar_epochs/"+subfolder+"/acc.png")
        plt.show()
        plt.close()

        # val accuracy
        for item in results:
            plt.plot(epochss, item['val_accuracy'] + [None for _ in range(max_epochs - len(item['val_accuracy']))])
        plt.legend([str(i) + " epochs" for i in self.epochs_list])
        plt.xlabel('epochs')
        plt.ylabel('val_accuracy')
        plt.title('Evolution of val_accuracy across epochs')
        plt.savefig("./analysis/compar_epochs/"+subfolder+"/val_acc.png")
        plt.show()
        plt.close()

        # loss
        for item in results:
            plt.plot(epochss, item['loss'] + [None for _ in range(max_epochs - len(item['loss']))])
        plt.legend([str(i) + " epochs" for i in self.epochs_list])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Evolution of loss across epochs')
        plt.savefig("./analysis/compar_epochs/"+subfolder+"/loss.png")
        plt.show()
        plt.close()

        # val loss
        for item in results:
            plt.plot(epochss, item['val_loss'] + [None for _ in range(max_epochs - len(item['val_loss']))])
        plt.legend([str(i) + " epochs" for i in self.epochs_list])
        plt.xlabel('epochs')
        plt.ylabel('val_loss')
        plt.title('Evolution of val_loss across epochs')
        plt.savefig("./analysis/compar_epochs/"+subfolder+"/val_loss.png")
        plt.show()
        plt.close()

        # accuracy / val_accuracy
        for i in range(len(results)):
            plt.plot(results[i]['epochs'], results[i]['accuracy'])
            plt.plot(results[i]['epochs'], results[i]['val_accuracy'])
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            title = "acc/val_acc for " + str(max(list(results[i]['epochs']))) + " epochs"
            plt.title(title)
            plt.legend(['acc', 'val_acc'])
            filename = "comp_acc_valacc_" + str(self.epochs_list[i]) + ".png"
            path = "./analysis/compar_epochs/"+subfolder+"/" + filename
            plt.savefig(path)
            plt.show()
            plt.close()

        # loss / val_loss
        for i in range(len(results)):
            plt.plot(results[i]['epochs'], results[i]['loss'])
            plt.plot(results[i]['epochs'], results[i]['val_loss'])
            plt.xlabel('epochs')
            plt.ylabel('loss')
            title = "loss/val_loss for " + str(max(list(results[i]['epochs']))) + " epochs"
            plt.title(title)
            plt.legend(['loss', 'val_loss'])
            filename = "comp_loss_valloss_" + str(self.epochs_list[i]) + ".png"
            path = "./analysis/compar_epochs/"+subfolder+"/" + filename
            plt.savefig(path)
            plt.show()
            plt.close()

    @staticmethod
    def analyze_predictions():
        # get epochs with corresponding prediction
        list_of_dirs = listdir("./histories")
        results = []
        for dirr in list_of_dirs:
            files = listdir(join('./histories', dirr))
            if files[0][-4:] == "json":
                file_to_open = files[0]
            else:
                file_to_open = files[1]
            with open("./histories/" + dirr + "/" + file_to_open, 'r') as file:
                data = json.load(file)
                results.append(
                    {
                        "epochs": data['epochs'],
                        "to_predict": data['to_predict'],
                        "prediction": data["prediction"]
                    }
                )

        # transform list of dict into dict and sort by id
        epochs_and_predictions = {}
        for i in range(len(results)):
            epochs_and_predictions[results[i]['epochs']] = results[i]['prediction']
        epochs_and_predictions = dict(collections.OrderedDict(sorted(epochs_and_predictions.items())))

        # display correctly epochs and corresponding predictions
        for i in range(len(list(epochs_and_predictions.items()))):
            buffer = list(epochs_and_predictions.items())[i]
            print(f"{buffer[0]} epochs - prediction: {buffer[1]} but expected: {results[0]['to_predict']}")

    @staticmethod
    def create_plots_figure(length_epochs_list):
        def save_fig_metrics(metric, data_obj, len_epochs):
            if metric == "accuracy":
                title = "acc / val_acc analysis"
                met, val_met = "accuracy", "val_accuracy"
                label = "Accuracy"
                path = "./analysis/compar_epochs/4350/acc_valacc_ana.png"
                x_pos_snd = 5
                y_pos_ths = 0.02
            else:
                title = "loss / val_loss analysis"
                met, val_met = "loss", "val_loss"
                label = "Loss"
                path = "./analysis/compar_epochs/4350/loss_valloss_ana.png"
                x_pos_snd = 3
                y_pos_ths = -0.5

            plt.figure(0)
            plt.suptitle(title)
            idx = 0
            for row in range(4):
                for col in range(3):
                    plt.subplot2grid((4, 3), (row, col))
                    plt.plot(data_obj[idx]["epochs"], data_obj[idx][met])
                    plt.plot(data_obj[idx]["epochs"], data_obj[idx][val_met], linestyle='--')
                    yy, locsy = plt.yticks()
                    lly = ['%.2f' % a for a in yy]
                    plt.yticks(yy, lly)
                    plt.grid()
                    plt.legend([met, val_met])
                    delta = abs(data_obj[idx][met][-1] - data_obj[idx][val_met][-1])
                    txt = r'$\Delta=' + str(delta) + "$"
                    text_x_pos = 2 + float(int(max(data_obj[idx]['epochs'])) * 7 / 80) if idx != 0 else x_pos_snd
                    plt.text(
                        text_x_pos,
                        float(max(data_obj[idx][met]) + y_pos_ths)
                        if metric == "loss"
                        else float(min(data_obj[idx][met]) + y_pos_ths),
                        txt,
                        fontsize=8
                    )
                    plt.scatter(max(data_obj[idx]['epochs']), data_obj[idx][met][-1], color='red', s=10)
                    plt.scatter(max(data_obj[idx]['epochs']), data_obj[idx][val_met][-1], color='red', s=10)
                    plt.plot(
                        [max(data_obj[idx]['epochs']), max(data_obj[idx]['epochs'])],
                        [data_obj[idx][met][-1], data_obj[idx][val_met][-1]],
                        color='red'
                    )
                    if row == 2 and col == 0:
                        plt.ylabel(label, rotation=90, fontsize=15)
                    if row == 3 and col == 0:
                        plt.xlabel("Epochs", fontsize=15)
                    if idx == len_epochs - 1:
                        break
                    else:
                        idx += 1

            figure = plt.gcf()
            figure.set_size_inches(14, 12)
            plt.savefig(path)

            print("==============================================")
            print(f"Figure saved in {path}")
            print("==============================================")

        # === END OF DEFINITION ===

        # get data
        list_of_dirs = listdir("./histories/4350")
        data = []
        # fig, axs = plt.figure(4, 4)
        for dirr in list_of_dirs:
            files = listdir(join("./histories/4350", dirr))
            if files[0][-4:] == "json":
                file_to_open = files[0]
            else:
                file_to_open = files[1]
            with open("./histories/4350/" + dirr + "/" + file_to_open, "r") as file:
                stocked = json.load(file)
                buffer = {
                    "epochs": [i + 1 for i in range(stocked['epochs'])],
                    "accuracy": stocked['accuracy'],
                    "val_accuracy": stocked['val_accuracy'],
                    "loss": stocked["loss"],
                    "val_loss": stocked['val_loss']
                }
                print(f"========== data in {file_to_open} ==========\n{buffer}")
                data.append(buffer)

        # sort data list of obj. by increasing epochs
        data = sorted(data, key=lambda k: k['epochs'])

        save_fig_metrics("accuracy", data, length_epochs_list)
        save_fig_metrics("loss", data, length_epochs_list)

    def fit_obs(self):
        # TODO
        pass

    def get_curve_state(last, penultimate):
        if last>penultimate:
            state="increasing"
        elif last<penultimate:
            state="decreasing"
        elif int(round(last, 0))==int(round(penultimate, 0)):
            state="stable"
        return state

    def confidence_percentage(li):
        if len(li[1])<len(li[0]):
            for i in range(len(li[0])-len(li[1])):
                li[1].append("...")
        eq = 0
        for exp, pre in zip(li[0], li[1]):
            if exp == pre:
                eq += 1
        return eq * 100 / len(li[0])
