class ReportWriter:
    def __init__(self):
        print("=== ReportWriter Initialized! ===")
        self.path = "./experimental_reports"
        self.filename = ""

    def write_first_page(self, filename, title, iter_data, params):
        self.filename = filename
        fp = ""
        fp += "# " + title + "\n"
        # iter on params
        fp += "\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
              ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
              ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
              ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
              ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Test on:\n "
        fp += "\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
              ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**" + str(
            list(iter_data.keys())[0]) + "**: " + str(iter_data[list(iter_data.keys())[0]]) + \
                    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
                   ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp" \
                   ";&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        fp += "\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**" + str(
            list(iter_data.keys())[1]) + "**: " + str(iter_data[list(iter_data.keys())[1]]) + "\n"
        # configuration
        fp += "# Parameters\n"
        for key in params:
            buffer = "\n* " + str(key) + ": " + str(params[key]) + "\n"
            fp += buffer
        # vocab infos
        fp += "\n# Vocab infos"
        fp += "\n\n|      Metric      | In vocab | Out vocab |"
        fp += "\n|:------------------:|:--------:|:---------:|"
        fp += "\n|Total nb. of words  |   354 148  |   247 994   |"
        fp += "\n|Nb. of unique words |   5 605   |   5 600    |"
        fp += "\n|Max seq. length     |     9    |     8     |"
        fp += "\n|Vocab size          |   5 603   |    5 599   |"
        fp += "\n|% no title          |   16.58%  |   16.66%   |\n"
        fp += "\n<br><br><br>\n"

        with open(self.path + "/" + self.filename, 'w') as md_file:
            md_file.write(fp)
            print("=====================================")
            print(f"{self.filename} saved in {self.path}")
            print("=====================================")

    def write_unique_epoch_page(self, gru_neuron, epoch,
                                acc_valacc_absolute_path, loss_valloss_absolute_path,
                                curves_data,
                                fitting_observations, predictions, bilan):

        content = ""
        content += "\n\n# GRU: " + str(gru_neuron) + " - Epochs: " + str(epoch)

        # learning curves
        content += "\n\n## Learning curves"
        # acc
        content += "\n\n![accuracy plot](" + str(acc_valacc_absolute_path) + ")\n"
        content += "\n\n<center>\n"
        content += "\n|          |      Training data      |  Validation data |   Delta   |"
        content += "\n|:--------:|:-----------------------:|:----------------:|:---------:|"
        content += "\n| minimum  | " + str(round(curves_data['accuracy']['tr_min'], 4)) + " | " + str(
            round(curves_data['accuracy']['val_min'], 4)) + " | " + str(
            round(abs(curves_data['accuracy']['tr_min'] - curves_data['accuracy']['val_min']), 4)) + " |"
        content += "\n| maximum  | " + str(round(curves_data['accuracy']['tr_max'], 4)) + " | " + str(
            round(curves_data['accuracy']['val_max'], 4)) + " | " + str(
            round(abs(curves_data['accuracy']['tr_max'] - curves_data['accuracy']['val_max']), 4)) + " |"
        content += "\n|   mean   | " + str(round(curves_data['accuracy']['tr_mean'], 4)) + " | " + str(
            round(curves_data['accuracy']['val_mean'], 4)) + " | " + str(
            round(abs(curves_data['accuracy']['tr_mean'] - curves_data['accuracy']['val_mean']), 4)) + " |"
        content += "\n|   state  | " + str(curves_data['accuracy']['tr_state']) + " | " + str(curves_data['accuracy']['val_state']) + " | " + "" + " |\n"
        content += "\n</center>\n\n"
        # loss
        content += "\n![loss plot](" + str(loss_valloss_absolute_path) + ")\n"
        content += "\n\n<center>\n\n"
        content += "\n|          |      Training data      |  Validation data |  Delta  |"
        content += "\n|:--------:|:-----------------------:|:----------------:|:-------:|"
        content += "\n| minimum  | " + str(round(curves_data['loss']['tr_min'], 4)) + " | " + str(
            round(curves_data['loss']['val_min'], 4)) + " | " + str(
            round(abs(curves_data['loss']['tr_min'] - curves_data['loss']['val_min']), 4)) + " |"
        content += "\n| maximum  | " + str(round(curves_data['loss']['tr_max'], 4)) + " | " + str(
            round(curves_data['loss']['val_max'], 4)) + " | " + str(
            round(abs(curves_data['loss']['tr_max'] - curves_data['loss']['val_max']), 4)) + " |"
        content += "\n|   mean   | " + str(round(curves_data['loss']['tr_mean'], 4)) + " | " + str(
            round(curves_data['loss']['val_mean'], 4)) + " | " + str(
            round(abs(curves_data['loss']['tr_mean'] - curves_data['loss']['val_mean']), 4)) + " |"
        content += "\n|   state  | " + str(curves_data['loss']['tr_state']) + " | " + str(curves_data['loss']['val_state']) + " | " + "" + " |\n"
        content += "\n\n</center>\n\n"

        # fitting osbervations
        content += "\n## Fitting observations"
        content += "\n\n<center>\n"
        content += "\n|          | Undefitting | Good fitting | Overfitting |"
        content += "\n|:--------:|:-----------:|:------------:|:-----------:|"
        content += "\n|  Status  | " + str(fitting_observations['under']) + " | " + str(
            fitting_observations['good']) + " | " + str(fitting_observations['over']) + " |\n"
        content += "\n\n</center>\n\n"

        # predictions
        content += "\n## Predictions"
        content += "\n\n<center>\n"
        content += "\n|          |          Seq. to predict           |          Expected seq.           |          " \
                   "Predicted seq.           | "
        content += "\n|:--------:|:----------------------------------:|:--------------------------------:|:---" \
                   "------------------------------:| "
        content += "\n|   Seq.   | " + str(predictions['to_predict']) + " | " + str(
            predictions['expected']) + " | " + str(predictions['predicted']) + " |\n"
        content += "\n\n</center>\n\n"

        # bilan
        content += "\n## Bilan"
        content += "\nFor **" + str(gru_neuron) + "** neurons in GRU and **" + str(
            epoch) + "** epochs, the training accuracy is gapped at **" + str(
            round(bilan['gap_acc'], 2)) + "**, the training loss at **" + str(
            round(bilan['gap_loss'], 2)) + "**. The model is **" + str(
            bilan['fit_status']) + "** and the prediction is correct at **" + str(
            round(bilan['pred_percent'], 2)) + "%**.\n"
        content += "<br><br>"

        with open(self.path + "/" + self.filename, 'a') as md_file:
            md_file.write(content)

    def write_last_page(self, dtlgth, gru_neuron):
        lp = ""

        # add model summary
        with open("./analysis/compar_epochs/" + str(dtlgth) + "/model_" + str(dtlgth) + ".h5", "r") as model_file:
            model = [line.strip('\n') for line in model_file]

        model_str = ""
        for mod in model:
            if "=" not in mod:
                model_str += mod
                model_str += "\n"

        lp += "\n# Model Summary\n"
        lp += model_str

        # add model plot
        lp += "<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>"
        lp += "\n# Plotted Model"
        # TODO
        # absolute_model_plot_path = "D:\\ERIS\dev_ia\\analysis\\compar_epochs\\" + str(dtlgth) + "\\model_" + str(dtlgth) + "_" + str(gru_neuron) + ".png"
        absolute_model_plot_path = "/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/"+str(dtlgth)+"/model_"+str(dtlgth)+"_"+str(gru_neuron)+".png"
        lp += "\n![plot_model](" + absolute_model_plot_path + ")"
        lp += "<br><br><br><br>"

        with open(self.path + "/" + self.filename, "a") as md_file:
            md_file.write(lp)
