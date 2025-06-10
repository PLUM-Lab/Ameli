# -*- coding: utf-8 -*-
"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.

"""
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def gen_dataset1():
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]
    return datasets



def gen_dataset4():
    product_json_path_str="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json" 
    product_json_path = Path(
        product_json_path_str
    )    
  
    X=[]
    y=[]
    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :
            review_json=review_dataset_json["reviews"][0]
            attribute_num=len(review_json["attribute"])
            is_low_quality_review=1 if review_json["is_low_quality_review"] else 0 
            image_similarity_score=review_json["image_similarity_score"]
            x_one_item=[attribute_num,image_similarity_score]
            y_one_item=is_low_quality_review
            X.append(x_one_item)
            y.append(y_one_item)
    return [[np.array(X),np.array(y)]]

def gen_dataset3():
    product_json_path_str="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json" 
    product_json_path = Path(
        product_json_path_str
    )    
  
    X=[]
    y=[]
    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :
            review_json=review_dataset_json["reviews"][0]
            attribute_num=len(review_json["attribute"])
            is_low_quality_review=1 if review_json["is_low_quality_review"] else 0 
            image_similarity_score=review_json["image_similarity_score"]
            x_one_item=[attribute_num,image_similarity_score]
            y_one_item=is_low_quality_review
            X.append(x_one_item)
            y.append(y_one_item)
    return [[np.array(X),np.array(y)]]

def gen_dataset2():
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
       
        make_circles(noise=0.2, factor=0.5, random_state=1),
        
    ]
    return datasets
from sklearn.metrics import classification_report, precision_recall_fscore_support


def train_classifier(args):
    X, y,product_category_list=gen_dataset8(args.factor,args.train_review_json_path)
    X_train, X_test,y_train, y_test,product_category_list_train, product_category_list_test = train_test_split(
        X, y, product_category_list,test_size=0.3, random_state=42
    )
    clf = make_pipeline(StandardScaler(), GaussianNB())
    clf.fit(X_train, y_train)
    return clf 

def gen_result(args):
    
     
    X, y,product_category_list=gen_dataset8(args.factor)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    result={}
    for name, clf in zip(names, classifiers):
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        y_pred =clf.predict(X_test)
        precision,recall, f1,_=precision_recall_fscore_support(y_test, y_pred, average='micro')
        print(classification_report(y_test,y_pred))
        c_report=classification_report(y_test, y_pred, output_dict=True)
        print(c_report)
        result[name]=[round(c_report["0"]["precision"],2),round(c_report["0"]["recall"],2), round(f1,2)] 
    print(result) 
    
def main():
    # datasets=gen_dataset()
    # X, y = ds
    datasets=gen_dataset3()
    X, y=datasets[0]
    figure = plt.figure(figsize=(27, 3))#27
    i = 1
    ds_cnt=0
    # iterate over datasets
    # for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
     
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

    plt.tight_layout()
    # plt.show()
    plt.savefig("bestbuy/output/detect.jpg")

import matplotlib.pyplot as plt
import numpy as np
def randrange(n, vmin, vmax):
        """
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        """
        return (vmax - vmin)*np.random.rand(n) + vmin
    
def gen_dataset5(ax):
    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, marker=m)
    return ax 

def gen_dataset6(ax):
    
    
    
    product_json_path_str="bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json" 
    product_json_path = Path(
        product_json_path_str
    )    
  
    X_dict={0:[],1:[]}
    y_dict={0:[],1:[]}
    z_dict={0:[],1:[]}
    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :
            review_json=review_dataset_json["reviews"][0]
            text_similarity_score=review_dataset_json["text_similarity_score"]
            product_title_similarity_score=review_dataset_json["product_title_similarity_score"]
            attribute_num=len(review_json["attribute"])
            is_low_quality_review=1 if review_json["is_low_quality_review"] else 0 
            image_similarity_score=review_json["image_similarity_score"]
            X_dict[is_low_quality_review].append(attribute_num)
            y_dict[is_low_quality_review].append(image_similarity_score)
            z_dict[is_low_quality_review].append(product_title_similarity_score)
             
    ax.scatter(X_dict[0], y_dict[0], z_dict[0], marker='o')
    ax.scatter(X_dict[1], y_dict[1], z_dict[1], marker='^')
    return ax 
    
def main3d():
    

    # Fixing random state for reproducibility
    # np.random.seed(19680801)


    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax=gen_dataset6(ax)

    ax.set_xlabel('attribute_num')
    ax.set_ylabel('image_similarity')
    ax.set_zlabel('text_similarity')

    # plt.show()
    plt.savefig("bestbuy/output/detect3D.jpg")
    
    
def gen_dataset7():
    iris = datasets.load_iris()
    X = iris.data[:, :3]  # we only take the first three features.
    Y = iris.target

    #make it binary classification problem
    X = X[np.logical_or(Y==0,Y==1)]
    Y = Y[np.logical_or(Y==0,Y==1)]
    return X,Y
    
def gen_dataset8(factor,product_json_path_str="bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json" ):
   
    product_json_path = Path(
        product_json_path_str
    )    
  
    X=[]
    y=[]
    product_category_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :
            review_json=review_dataset_json["reviews"][0]
            text_similarity_score=review_dataset_json["text_similarity_score"]
            product_title_similarity_score=review_dataset_json["product_title_similarity_score"]
            attribute_num=len(review_json["attribute"])
            product_category=review_dataset_json["product_category"]
            is_low_quality_review=1 if review_json["is_low_quality_review"] else 0 
            image_similarity_score=review_json["image_similarity_score"]
            if factor=="3":
                x_one_item=[attribute_num,image_similarity_score,product_title_similarity_score]
            elif factor=="4":
                x_one_item=[attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score]
            else:
                x_one_item=[attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score]
                product_category_list.append(product_category)
            y_one_item=is_low_quality_review
            X.append(x_one_item)
            y.append(y_one_item)
    return  np.array(X),np.array(y) ,product_category_list

 
             
    
def main3d2():
    from sklearn.svm import SVC
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets
    from mpl_toolkits.mplot3d import Axes3D

    X,Y=gen_dataset8()

    model = svm.SVC(kernel='linear')
    clf = model.fit(X, Y)

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

    tmp = np.linspace(0,8,8)
    tmp2 = np.linspace(0,1,10)
    x,y = np.meshgrid(tmp,tmp2)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
    ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
    # ax.plot_surface(x, y, z(x,y))
    ax.view_init(30, 60)
    # plt.show()
    plt.savefig("bestbuy/output/detect3D2.jpg")
    
def rule1(attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score):
    if attribute_num<=1 and image_similarity_score<0.7 and text_similarity_score<1:
        return 1
    else:
        return 0


def rule2(attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category):
    if attribute_num<=attribute_num_threshold and image_similarity_score<=image_similarity_threshold and text_similarity_score<=text_similarity_threshold and product_title_similarity_score<=product_title_similarity_threshold :
        return 1
    else:
        return 0

def check_mode(mode,factor_list,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold):
    if mode=="and":
        if "attribute" in factor_list:
            if attribute_num>attribute_num_threshold:
                return 0
        if "image" in factor_list:
            if image_similarity_score>image_similarity_threshold :
                return 0
        if "title" in factor_list:
            if product_title_similarity_score >product_title_similarity_threshold:
                return 0
        if "desc" in factor_list:
            if text_similarity_score >text_similarity_threshold :
                return 0
        return 1 
    else:
        if "attribute" in factor_list:
            if attribute_num<=attribute_num_threshold:
                return 1
        if "image" in factor_list:
            if image_similarity_score<=image_similarity_threshold :
                return 1
        if "title" in factor_list:
            if product_title_similarity_score <=product_title_similarity_threshold:
                return 1
        if "desc" in factor_list:
            if text_similarity_score<=text_similarity_threshold :
                return 1
        return 0 

def rule4(factor_list    ,mode,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category):
    if "Best Buy -> TV & Home Theater -> TVs" in product_category :#and attribute_num<5
        return 1 
    return check_mode(mode,factor_list,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold)


def rule5(factor_list    ,mode,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category):
  
    for filter_category in filter_category_list:
        if filter_category  in product_category :#and attribute_num<5
            return 1 
    return check_mode(mode,factor_list,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold)



def rule3(attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category):
    if "Best Buy -> TV & Home Theater -> TVs" in product_category and attribute_num<5:
        return 1 
    elif attribute_num<=attribute_num_threshold and image_similarity_score<=image_similarity_threshold and text_similarity_score<=text_similarity_threshold and product_title_similarity_score<=product_title_similarity_threshold :
        return 1
    else:
        return 0    

def clean_by_rule(args,  attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold ):    
    X, y,product_category_list=gen_dataset8(args.factor,args.review_json_path)
     
    result={}
    y_pred=[]
    for x_item,product_category in zip(X,product_category_list):
        attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score=x_item
        # product_title_similarity_score=100
        y_pred_item=rule3(attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category)
        y_pred.append(y_pred_item)
 
    precision,recall, f1,_=precision_recall_fscore_support(y, y_pred, average='micro')
    print(classification_report(y,y_pred))
    # c_report=confusion_matrix(y, y_pred, output_dict=True)
    # print(c_report)

def check_one_model( args,factor_list,mode,X,y,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category_list):
    y_pred=[]
    for x_item,product_category in zip(X,product_category_list):
        attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score=x_item
        y_pred_item=rule4(factor_list,mode,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,
                          attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,
                          product_title_similarity_threshold,product_category)
        y_pred.append(y_pred_item)
    if args.mode in["filter_category_train_test","filter_category_test"]:
        y_pred=update_y_pred_by_filter_category(X,y_pred,product_category_list)
    precision,recall, f1,_=precision_recall_fscore_support(y, y_pred, average='micro')
    score=f1
    c_report=classification_report(y, y_pred, output_dict=True)
    # score=c_report["1"]["recall"]
    # score=c_report["1"]["f1-score"]
    # valid_item_score=c_report["0"]["recall"]
    # if valid_item_score>=0.80:
    #     score=c_report["0"]["precision"]
    valid_item_score=c_report["0"]["precision"]
    if valid_item_score>=0.85:
        score=c_report["0"]["recall"]
        print(classification_report(y,y_pred))
        return score,y_pred
    else:
        return 0,None 
     
          
def hypersearch(args):
    X, y,product_category_list=gen_dataset8(args.factor,args.train_review_json_path)
    if args.testset=="all":
        X_test,y_test, product_category_list_test =    X, y, product_category_list
    else:
        X_train, X_test,y_train, y_test,product_category_list_train, product_category_list_test = train_test_split(
            X, y, product_category_list,test_size=0.3, random_state=42
        )
    
    max_score=-1
    y_pred_max=[]
    max_threshold=[]
    for factor_list in [["attribute","image"],["attribute","title"],["attribute","desc"],["attribute","image","title"],["attribute","image","desc"],["attribute","desc","title"],["attribute","image","desc","title"]]:#
        for mode in ["and"]:#"and",,"or"
            for attribute_num_threshold in [1,2,3,4]:
                for image_similarity_threshold in [0.5,0.6,0.7,0.8,0.9]:
                    for text_similarity_threshold in [-7.5,-5,-2.5,0,2.5,5]:
                        for product_title_similarity_threshold in [-7.5,-5,-2.5,0,2.5,5]:
                            score,y_pred=check_one_model(args,factor_list,mode,X_test,y_test,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category_list_test)
                            if score>max_score:
                                max_score=score 
                                y_pred_max=y_pred 
                                max_threshold=[attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,factor_list,mode]
    print(f"{max_score}, {max_threshold}")
    print(classification_report(y_test,y_pred_max))
    print(confusion_matrix(y_test,y_pred_max))
    
    # c_report=classification_report(y, y_pred_max, output_dict=True)
    # print(c_report)
    
    
def show_clean(args,factor_list,mode,product_json_path_str,output_products_path_str,factor,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,is_save=True):
    # product_json_path_str="bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json" 
    product_json_path = Path(
        product_json_path_str
    )    
    # clf=train_classifier(args)
    X=[]
    y=[]
    output_products_path = Path(
        output_products_path_str#f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v3_predict.json'
    ) 
    valid_num=0
    out_list=[]
    y_pred=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :
            review_json=review_dataset_json["reviews"][0]
            text_similarity_score=review_dataset_json["text_similarity_score"]
            product_category=review_dataset_json["product_category"]
            product_title_similarity_score=review_dataset_json["product_title_similarity_score"]
            # product_title_similarity_score=100
            attribute_num=len(review_json["attribute"])
            if "is_low_quality_review" in review_json:
                is_low_quality_review=1 if review_json["is_low_quality_review"] else 0 
                y.append(is_low_quality_review)
            image_similarity_score=review_json["image_similarity_score"]
            if factor=="3":
                x_one_item=[attribute_num,image_similarity_score,product_title_similarity_score]
            else:
                x_one_item=[attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score]    
            y_pred_item=rule5(factor_list,mode,attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold,product_category)
            review_dataset_json["predicted_is_low_quality_review"]=True if y_pred_item==1 else False 
            if y_pred_item==0:
                out_list.append(review_dataset_json)
            y_pred.append(y_pred_item)
            if y_pred_item==0:
                valid_num+=1
        if is_save:
            with open(output_products_path, 'w', encoding='utf-8') as fp:
                json.dump(out_list, fp, indent=4)          
    if len(y_pred)==len(y):
        print(classification_report(y,y_pred))
        print(confusion_matrix(y,y_pred)) 
    print(f"valid: {valid_num}")        
            
            
            

    
def show_clean_by_nb(args,factor_list,mode,product_json_path_str,output_products_path_str,factor,attribute_num_threshold,image_similarity_threshold,text_similarity_threshold,product_title_similarity_threshold):
    product_json_path = Path(
        product_json_path_str
    )    
    clf=train_classifier(args)
    X=[]
    y=[]
    product_category_list=[]
    output_products_path = Path(
        output_products_path_str#f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v3_predict.json'
    ) 
    valid_num=0
    out_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :
            review_json=review_dataset_json["reviews"][0]
            text_similarity_score=review_dataset_json["text_similarity_score"]
            product_category=review_dataset_json["product_category"]
            product_title_similarity_score=review_dataset_json["product_title_similarity_score"]
            # product_title_similarity_score=100
            attribute_num=len(review_json["attribute"])
            if "is_low_quality_review" in review_json:
                is_low_quality_review=1 if review_json["is_low_quality_review"] else 0 
            image_similarity_score=review_json["image_similarity_score"]
            if factor=="3":
                x_one_item=[attribute_num,image_similarity_score,product_title_similarity_score]
            elif factor=="4":
                x_one_item=[attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score]    
            else:
                x_one_item=[attribute_num,image_similarity_score,text_similarity_score,product_title_similarity_score]    
            product_category_list.append(product_category)
                
                
            X.append(x_one_item)
            # y.append(is_low_quality_review)
           
          
            # if y_pred_item==0:
            #     valid_num+=1
                
        y_pred=clf.predict(X)
        if args.mode in["filter_category_train_test","filter_category_test"]:
            y_pred=update_y_pred_by_filter_category(X,y_pred,product_category_list)
        for idx,review_dataset_json   in  enumerate(new_crawled_products_url_json_array) :
            if y_pred[idx]==1 :
                
                review_dataset_json["predicted_is_low_quality_review"]=True   
            else:
                review_dataset_json["predicted_is_low_quality_review"]=False
                valid_num+=1
            out_list.append(review_dataset_json)
        with open(output_products_path, 'w', encoding='utf-8') as fp:
            json.dump(out_list, fp, indent=4)          
    print(f"valid: {valid_num}")              
            
            

filter_category_list=["Best Buy -> Computers & Tablets -> Monitors",
"Best Buy -> TV & Home Theater -> TVs",
"Best Buy -> Computers & Tablets -> Computer Cards & Components -> Motherboards",
"Best Buy -> Computers & Tablets -> Computer Cards & Components -> Fans, Heatsinks & Cooling ",
"Best Buy -> Cameras, Camcorders & Drones -> Digital Cameras",
"Best Buy -> Cameras, Camcorders & Drones -> Digital Camera Accessories -> Camera Lenses",
"Best Buy -> Computers & Tablets -> Computer Cards & Components -> GPUs / Video Graphics Cards",
"Best Buy -> TV & Home Theater -> Projectors & Screens",
"Best Buy -> Audio -> Home Audio -> Speakers -> Sound Bars",
"Best Buy -> Audio -> Home Audio Accessories -> Speaker Accessories -> Speaker Stands & Mounts -> Sound Bar Mounts",
"Best Buy -> Computers & Tablets -> PC Gaming -> Gaming Monitors"]
            
            

def filter_by_category(X,y,product_category_list):
    new_X,new_y,new_product_category_list=[],[],[]
    for x_item,y_item,product_category in zip(X,y,product_category_list):
        is_filter=False 
        attribute_num=x_item[0]
        for filter_category in filter_category_list:
            if filter_category  in product_category :#and attribute_num<5
                is_filter=True 
        if not is_filter:
            new_X.append(x_item)
            new_y.append(y_item)
            new_product_category_list.append(product_category)
    print(f"remain {len(new_X)}")
    return np.array(new_X),np.array(new_y) ,new_product_category_list

def update_y_pred_by_filter_category(X,y_pred,product_category_list_test):
    new_y =[] 
    for x_item,y_item,product_category in zip(X,y_pred,product_category_list_test):
        is_filter=False 
        attribute_num=x_item[0]
        for filter_category in filter_category_list:
            if filter_category  in product_category :#and attribute_num<5
                is_filter=True 
        if not is_filter:
           
            new_y.append(y_item)
        else:
            new_y.append(1)
      
   
    return  np.array(new_y)  

def train_test_classifier (args):
    X, y,product_category_list=gen_dataset8(args.factor,args.train_review_json_path)
    X_train, X_test,y_train, y_test,product_category_list_train, product_category_list_test = train_test_split(
        X, y, product_category_list,test_size=0.3, random_state=42
    )
    if args.mode=="filter_category_train_test":
        X_train,y_train,product_category_list_train=filter_by_category(X_train,y_train,product_category_list_train)
  
    
    result={}
    for name, clf in zip(names, classifiers):
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        y_pred =clf.predict(X_test)
        if args.mode in["filter_category_train_test","filter_category_test"]:
            y_pred=update_y_pred_by_filter_category(X_test,y_pred,product_category_list_test)
        precision,recall, f1,_=precision_recall_fscore_support(y_test, y_pred, average='micro')
        print(classification_report(y_test,y_pred))
        print(confusion_matrix(y_test,y_pred ))
        c_report=classification_report(y_test, y_pred, output_dict=True)
        print(c_report)
        result[name]=[round(c_report["0"]["precision"],2),round(c_report["0"]["recall"],2), round(f1,2)] 
    print(f"{len(X_train)},{len(X_test)}")
    print(result) 
            
def clean_dataset_by_classifier(product_json_path_str,output_products_path_str):
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str#f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v3_predict.json'
    ) 
    out_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in  new_crawled_products_url_json_array :             
            if not review_dataset_json["predicted_is_low_quality_review"]:
                out_list.append(review_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
            
            
            
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,help=" ",default="filter_category_test") #filter_category_train_test,filter_category_test
    parser.add_argument('--factor',type=str,help=" ",default="5")
    # parser.add_argument('--metric',type=str,help=" ",default="acc")
    parser.add_argument('--train_review_json_path',type=str,help=" ",default="bestbuy/data/example/bestbuy_100_human_performance_500_clean.json")
    # parser.add_argument('--dataset_review_json_path',type=str,help=" ",default="bestbuy/data/final/v2_course/bestbuy_review_2.3.16.14_round2.json" )#bestbuy/data/final/v2_course/bestbuy_review_2.3.16.15_filter_low_information.json
    # parser.add_argument('--dataset_output_path',type=str,help=" ",default="bestbuy/data/final/v2_course/bestbuy_review_2.3.16.15_filter_low_information.json")#bestbuy/data/final/v2_course/bestbuy_review_2.3.16.16_filter_low_information.json
    parser.add_argument('--dataset_review_json_path',type=str,help=" ",default="bestbuy/data/final/v2_course/bestbuy_review_2.3.16.15_filter_low_information.json")
    parser.add_argument('--dataset_output_path',type=str,help=" ",default="bestbuy/data/final/v2_course/bestbuy_review_2.3.16.16_filter_low_information.json")
    parser.add_argument('--testset',type=str,help=" ",default="all")
    args = parser.parse_args()
    return args

  
  
  
if __name__ == '__main__':
    args = parser_args()
    rule_mode="and"
    factor_list=["attribute","image","desc","title"]#"desc", ,"title"
    # train_test_classifier(args)
    # show_clean_by_nb(args,factor_list,rule_mode,args.dataset_review_json_path,args.dataset_output_path,args.factor, 1, 0.7 , 5, 0)
    
    
    # hypersearch(args)
    show_clean(args,factor_list,rule_mode,args.dataset_review_json_path,args.dataset_output_path,args.factor, 3, 0.7, 0, 2.5)#3, 0.7, -7.5, 2.5
    # clean_dataset_by_classifier(args.dataset_review_json_path,args.dataset_output_path)
    
    
    
    # show_clean(args,factor_list,rule_mode,args.dataset_review_json_path,args.dataset_output_path,args.factor,4, 0.7, 0, 0,is_save=False)#3, 0.7, -7.5, 2.5
         