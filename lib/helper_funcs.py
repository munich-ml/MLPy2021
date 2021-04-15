# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:04:44 2020

@author: holge
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


def pickle_in(filename, scope):
    """
    Pickles in variables from file and adds them to the scope.  
    
    Example usage:
        pickle_in("fn.pkl", locals())

    Parameters:
        filename : str
    
        scope : dict
            usually locals(), globals() works, too

    Returns:
        None
    """
    with open(filename, 'rb') as file:
        d = pickle.load(file)
        print("pickle_in: Updating scope with the following variables:")
        print(list(d.keys()))
        scope.update(d)


def pickle_out(filename, scope, *variables):
    """
    Pickles out all *variables (including their names) 
    You are required to put a scope e.g. locals() as argument.
    
    Example usage:
        pickle_out("fn.pkl", locals(), X_test, y_test)

    Parameters:
        filename : str
    
        scope : dict
            usually locals()
            globals() works, too
            
        *variables : all types that can be pickled
            the variables to be pickled

    Returns
        None
    """
    # step 1: Create a dict of vars to be pickled out
    #         The names are gathered from the scope (e.g. locals())
    d = dict()
    for name, val in scope.items():
        for var in variables:
            if var is val:
                d[name] = val
    
    # step 2: dump to file    
    with open(filename, 'wb') as file:
        pickle.dump(d, file)
        
        
def parse_logfile_string(s):
    """
    Parses a logfile string according to 10_Logfile_challenge.ipynb

    Parameters:
        s : str
            logfile string

    Returns:
        dictionary : dict
            containing "params", "names", "data"
    """
    # split the input string on "\n" new line
    lines = s.split("\n")

    # create a look-up table of sections and line numbers
    idxs = dict()
    for lineNo, line in enumerate(lines):
        if line in ['measurements', "header"]:
            idxs[line] = lineNo 
    idxs["names"] = idxs["measurements"] + 1
    idxs["params_begin"] = idxs["header"] + 1
    idxs["params_end"] = idxs["measurements"] - 1
    idxs["data"] = idxs["names"] + 1

    # parse the column 
    names = lines[idxs["names"]].split(",")

    # parse the params_lines list(str) into params dict{param: value}
    params = dict()
    for line in lines[idxs["params_begin"] : idxs["params_end"]]:
        key, value = line.split(",")
        params[key] = value

    # converts str to float incl. "Ohms" removal
    def string_to_float(s):
        idx = s.find("Ohms")
        if idx > 0:
            number = s.split(" ")[0]
            prefix = s[idx-1]
            return float(number) * {" ": 1, "m": 0.001}[prefix]
        return float(s)

    # parse data_lines list(str) into data list(list(floats))
    data = list()
    for data_line in lines[idxs["data"] :]:
        row = list()
        for item in data_line.split(","):
            row.append(string_to_float(item))
        data.append(row)

    return {"params": params, "names": names, "data":data}


def create_csv(plot=False):
    """
    creates a "logfile.csv" in the datasets subdir to be used in the logfile challenge.
    """
    np.random.seed(23)
    
    def string(x, make_ohms=False):
        if type(x) in (float, np.float64):
            if make_ohms:
                if(abs(x) < 1):
                    return "{:.0f} mOhms".format(1000*x)
                return "{:.2f} Ohms".format(x)
            return "{:.2f}".format(x)
        return str(x).replace(",", "_")
    
    
    # Create header
    cal_factors = [0.55, 1, 1.88]
    
    header = {"measurement date": dt.date(2021, 4, 6),
              "measurement time": dt.time(8, 0, 0)}
    
    for i, cal_factor in enumerate(cal_factors):
        if cal_factor != 1:
            header["calibration factor sig{}".format(i)] = cal_factor
    
    s = "MLPy logfile challenge"
    
    s += "\n\nheader"
    for key, value in header.items():
        s += "\n" + string(key) + "," + string(value)
    
    # Create content
    s += "\n\nmeasurements"
    s += "\nx,sig0,sig1,sig2"
    
    x = 10 * np.random.rand(30)    # sig x-axis
    signals = [x]

    for i, (sig_period, cal_factor) in enumerate(zip([6, 8, 10], cal_factors)):
        signal = 10 * (1 - np.cos(2*np.pi* x / sig_period))
        signals.append(signal / cal_factor)
        if plot:
            plt.plot(x, signal, "x", label="sig"+str(i))
    
    if plot:
        plt.xlabel("x"), plt.ylabel("sig [ohms]")
        plt.grid(), plt.legend()

    # write lines
    for values in zip(*signals):
        newline = "\n"
        for val in values:
            make_ohms = val != values[0]
            newline += string(val, make_ohms) + ","
        s += newline[:-1]
    
    # store file to datasets folder
    folder_name = "datasets"
    if folder_name not in os.listdir():
        os.mkdir(folder_name)

    with open(os.path.join(folder_name, "logfile.csv"), "w") as file:
        file.write(s)


def plot_confusion_matrix(cm, xticks, yticks, normalize=False, ignore_main_diagonal=False, 
                          cmap=plt.cm.binary):
    """
    plots a confusion matrix using matplotlib

    Parameters
    ----------
    cm : (tensor or numpy array)
        confusion matrix
        e.g. from tf.math.confusion_matrix(y_test, y_pred)
    xticks : (list)
        x tick labels
    yticks : (list)
        x tick labels
    normalize : (bool), optional
        scales cm to 1. The default is False.
    ignore_main_diagonal : (bool), optional
        sets the main diagonal to zero. The default is False.
    cmap : matplotlib colormap, optional
        

    Returns
    -------
    None.

    """
    
    cm = np.array(cm)
    if normalize:   # normalize to 1.0
        cm = cm / cm.max()
    if ignore_main_diagonal:  # set main diagonal to zero
        for i in range(len(cm)):
            cm[i, i] = 0
    plt.imshow(cm, cmap=cmap)
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=90)
    plt.yticks(ticks=range(len(yticks)), labels=yticks)
    plt.xlabel("predicted class")
    plt.ylabel("actual class")
    # put numbers inside the heatmap
    thresh = cm.max() / 2.
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            plt.text(j, i, format(int(val)),
                    horizontalalignment="center",
                    color = "white" if val > thresh else "black")
                    
    plt.colorbar()
    
    
def plot_prediction_examples(test_class, class_names, y_pred, y_test, X_test, 
                             n_cols=10, cmap=plt.cm.binary):
    """
    plots images of predictions examples in 4 rows from 'true positives' till 'false negatives'

    Parameters
    ----------
    test_class : TYPE
        DESCRIPTION.
    class_names : TYPE
        DESCRIPTION.
    y_pred : np.array
        predicted classes
    y_test : np.array
        true / actual classes
    X_test : np.array
        array of images
    n_cols : int, optional
        Number of columns / number of images per row. The default is 10.
    cmap : matplotlib colormap, optional

    Returns
    -------
    None.

    """
    
    print("Evaluating examples of test_class={}, '{}'".format(test_class, 
                                                              class_names[test_class]))
    
    # step 1: Compute TP, TN, FP, FN
    preds = {"true pos": [],
             "true neg": [],
             "false pos": [],
             "false neg": []}

    for i, (val_test, val_pred) in enumerate(zip(y_test, y_pred)):
        if val_test == test_class:
            if val_pred == test_class:
                preds["true pos"].append((i, val_test, val_pred))
            else:
                preds["false neg"].append((i, val_test, val_pred))
        else:
            if val_pred == test_class:
                preds["false pos"].append((i, val_test, val_pred))
            else:
                preds["true neg"].append((i, val_test, val_pred))

    for key, val in preds.items():
        print("- {}: {} images".format(key, len(val)))
    
    # step 2: plotting random examples of right and wrong predictions
    plt.figure(figsize=(n_cols*1.8, 9))
    for row, predictions in enumerate(preds.values()):
        for col, idx in enumerate(np.random.randint(0, len(predictions), n_cols)):
            i, val_test, val_pred = predictions[idx]
            plt.subplot(len(preds), n_cols, n_cols*row+col+1)
            plt.imshow(np.squeeze(X_test[i]), cmap=cmap)
            plt.axis('off')
            title = "\nimage:{}\nact: {}\nprd: {}".format(i, class_names[val_test], class_names[val_pred])
            plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_hist_2D(df, x_column, y_column, bins=15, levels=20, figsize=[13, 4], cmap=plt.cm.coolwarm):
    """
    Parameters
    ----------
    df : Pandas DataFrame
        Data
    x_column : str
        column to plot on x
    y_column : str
        column to plot on y
    bins : int, optional
        Number of histogram bins. The default is 15.
    levels : int, optional
        Number of contour levels. The default is 20.
    figsize : tuple or list, optional
        The default is [13, 4].
    cmap : matplotlib colormap, optional

    Returns
    -------
    None

    """
    x = df[x_column]
    y = df[y_column]
    fig = plt.figure(figsize=figsize) 
    axes = fig.subplots(nrows=1, ncols=2)
    cnts, h2x, h2y, img = axes[0].hist2d(x, y, bins=bins, cmap=cmap,
                                         range=([x.min(), x.max()], [y.min(), y.max()]))
    axes[0].set_title("hist2d heatmap")

    def edges2centers(edges):
        return (edges[1:] + edges[:-1]) / 2

    axes[1].contourf(edges2centers(h2x), edges2centers(h2y), cnts.T, levels=levels, 
                     cmap=cmap)
    axes[1].set_title("contour plot")

    for ax in axes:
        ax.grid(which="both")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        fig.colorbar(img, ax=ax)
    
    fig.tight_layout()
    

if __name__ == "__main__":
    # logfile challenge, create csv
    create_csv()

    # Test plot_hist_2D
    x = np.random.beta(a=2, b=5, size=10000)
    y = np.random.beta(a=1.5, b=3, size=10000)
    df = pd.DataFrame(np.column_stack([x, x+y]), columns=["x", "y"])
    plot_hist_2D(df, "x", "y", bins=20)
    