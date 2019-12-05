from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import pandas as pd
import os
from helper import *
import matplotlib.pyplot as plt


# data_frame = load_pickle('../Data/df_for_metric.pkl')
# #data_frame.head()
# data_frame = data_frame.iloc[:,2:]

cf_list = load_pickle('../Output/loss_fn.pkl')
data_frame_predict, data_frame_true = load_pickle('../Output/test_output_transcript_only.pkl')
data_frame_predict =convert_dict_to_categorical(data_frame_predict)
data_frame_true =convert_dict_to_categorical(data_frame_true)


# print(data_frame_predict)
# print(data_frame_true)

rating_names = ['beautiful', 'confusing', 'courageous', 'fascinating', 'funny', 'informative', 'ingenious', 'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok', 'persuasive', 'unconvincing']
protected_attribute_maps = [{0.0: 'male', 1.0: 'female',2.0:'gender_other'}]
default_mappings = {
    'label_maps': [{1.0: 1, 0.0: 0}],
    'protected_attribute_maps': [{0.0: 'white', 1.0: 'black_or_african_american',2.0:'asian',3.0:'race_other'},
                                 {0.0: 'male', 1.0: 'female',2.0:'gender_other'}]
}

prot_attr_dict = {'race':{0.0: 'white', 1.0: 'black_or_african_american',2.0:'asian',3.0:'race_other'},
                                 'gender':{0.0: 'male', 1.0: 'female',2.0:'gender_other'}}

privileged_classes=[lambda x: x == 'white',lambda x: x == 'male']
#privileged_classes=['white','male']
protected_attribute_names=['race', 'gender']
unpriv_list = [[{'gender':0,'race':1},{'gender':0,'race':2},{'gender':0,'race':3}],
    [{'gender':1,'race':1},{'gender':1,'race':2},{'gender':1,'race':3}],
    [{'race':1},{'race':2},{'race':3}],
    [{'gender':1},{'gender':2}],
    [{'gender':1,'race':1},{'gender':2,'race':1}]]
priv_list = [[{'gender':0,'race':0}],
    [{'gender':1,'race':0}],
    [{'race':0}],
    [{'gender':0}],
    [{'gender':0,'race':1}]]


#Data_set_list = [(,)]
# predict_df_list = [pd.DataFrame({}) for i in range(14)]
# true_df_list = [pd.DataFrame({}) for i in range(14)]

# for col in data_frame_true.columns:
#     if col in rating_names:
#         ind =  rating_names.index(col)
#         predict_df_list[ind][[col]] = data_frame_predict[[col]]
#         true_df_list[ind][[col]] = data_frame_true[[col]]
#     else:
#         for i in range(14):
#             predict_df_list[i][[col]] = data_frame_predict[[col]]
#             true_df_list[i][[col]] = data_frame_true[[col]]




def plot_using_aif(df_predict,df_true):
   

    predict_list, true_list = [], []
    unpriv_label_list , priv_label_list = [], []
    for (u,p) in zip(unpriv_list,priv_list):
        cur_predict, cur_true = [], []

        unpriv_label = '+'.join(['-'.join([prot_attr_dict[key][u_el[key]] for key in u_el]) for u_el in u])
        priv_label = '+'.join(['-'.join([prot_attr_dict[key][p_el[key]] for key in p_el]) for p_el in p])

        print('-------------------------------------------------------------------')
        print('unpriv_label:-->',unpriv_label)
        print('-------------------------------------------------------------------')
        print('priv_label  :-->',priv_label)
        print('-------------------------------------------------------------------')
        print('\n\n')
        for i,label in enumerate(rating_names):
            #print('Fairness Metric for the label------>',label.upper())
        
            predict_dataset  = StandardDataset(df=predict_df_list[i], label_name=label, favorable_classes=[1.0,1.0],
                                protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes) 
            
            true_dataset  = StandardDataset(df=true_df_list[i], label_name=label, favorable_classes=[1.0,1.0],
                                protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes) 
            
           
            predict_dataset_metric = BinaryLabelDatasetMetric(predict_dataset, unprivileged_groups=u, privileged_groups=p)
            true_dataset_metric = BinaryLabelDatasetMetric(true_dataset, unprivileged_groups=u, privileged_groups=p)
            

            #classfication_metric = ClassificationMetric(true_dataset, predict_dataset, unprivileged_groups=u, privileged_groups=p)
            
            #x=classfication_metric.generalized_entropy_index()
            
            #print(label,':  -->','predicted :  -->',abs(predict_dataset_metric.disparate_impact()),'true :  -->',abs(true_dataset_metric.disparate_impact()))
            print(label,':  -->','predicted :  -->',abs(predict_dataset_metric.mean_difference()),'true :  -->',abs(true_dataset_metric.mean_difference()))

            # cur_predict.append(abs(predict_dataset_metric.mean_difference()))
            # cur_true.append(abs(true_dataset_metric.mean_difference()))

        # predict_list.append(cur_predict)
        # true_list.append(cur_true)
            #print("For Rating",label,"Difference in mean outcomes between unprivileged and privileged groups = %f" %diff)

        
         

        
    #     unpriv_label_list.append(unpriv_label)
    #     priv_label_list.append(priv_label)

    # predict_list = np.array(predict_list)
    # true_list = np.array(true_list)

    # for i in range(len(unpriv_list)):

    #     cur_predict = predict_list[i]
    #     cur_true = true_list[i]

    #     xaxis = np.arange(14)
    #     width = 0.3

    #     fig, ax = plt.subplots()
    #     rects1 = ax.bar(xaxis - width/2, cur_predict, width, label='after')
    #     rects2 = ax.bar(xaxis + width/2, cur_true, width, label='before')

    #     # Add some text for labels, title and custom x-axis tick labels, etc.
    #     ax.set_ylabel('Group Probability before running classifier')
    #     ax.set_title('Group Fairness for different Label')
    #     ax.set_xticks(xaxis)
    #     ax.set_xticklabels(rating_names,rotation=90)
    #     ax.legend()

    #     plt.show()
        




    # a_list_true, b_list_true = [], []
    # unpriv_label_list , priv_label_list = [], []
    # for (u,p) in zip(unpriv_list,priv_list):
    #     cur_a, cur_b = [], []
    #     for label in rating_names:
    #         #print('Fairness Metric for the label------>',label.upper())
        
    #         dataset  = StandardDataset(df=df_true, label_name=label, favorable_classes=[1.0,1.0],
    #                             protected_attribute_names=protected_attribute_names, privileged_classes=privileged_classes) 
            
           
    #         dataset_metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=u, privileged_groups=p)
            


    #         diff = dataset_metric.mean_difference()
    #         ratio = dataset_metric.disparate_impact()
    #         a,b = ab_ret(diff,ratio)
    #         cur_a.append(a)
    #         cur_b.append(b)

    #     a_list_true.append(cur_a)
    #     b_list_true.append(cur_b)
    #         #print("For Rating",label,"Difference in mean outcomes between unprivileged and privileged groups = %f" %diff)

        
         

    #     unpriv_label = '+'.join(['-'.join([prot_attr_dict[key][u_el[key]] for key in u_el]) for u_el in u])
    #     priv_label = '+'.join(['-'.join([prot_attr_dict[key][p_el[key]] for key in p_el]) for p_el in p])

    #     unpriv_label_list.append(unpriv_label)
    #     priv_label_list.append(priv_label)

    # a_list_true = np.array(a_list_true)
    # b_list_true = np.array(b_list_true)


    # for i in range(14):

    #     x= a_list_predict[:,i]
    #     y= b_list_predict[:,i]

    #     z= a_list_true[:,i]
    #     w= b_list_true[:,i]

    #     plt.subplot(3,5,i+1)
    #     plt.scatter(x,y,color='blue')
    #     plt.scatter(z,w,color='red',label=rating_names[i])

    #     for j in range(len(x)):
            
    #         plt.arrow(z[j],w[j],x[j]-z[j],y[j]-w[j],length_includes_head=True,head_width=0.03)

    #     min_range = min(min(x),min(y),min(z),min(w))
    #     max_range = max(max(x),max(y),max(z),max(w))
    #     xax = np.linspace(min_range,max_range,100)
    #     plt.plot(xax,xax)

    #     #plt.legend()
        

    # plt.show()

    # for i in range(len(a_list_true)):

    #     xaxis = np.arange(14)
    #     width = 0.3

    #     fig, ax = plt.subplots()
    #     rects1 = ax.bar(xaxis - width/2, a_list_true[i], width, label=unpriv_label_list[i])
    #     rects2 = ax.bar(xaxis + width/2, b_list_true[i], width, label=priv_label_list[i])

    #     # Add some text for labels, title and custom x-axis tick labels, etc.
    #     ax.set_ylabel('Group Probability before running classifier')
    #     ax.set_title('Group Fairness for different Label')
    #     ax.set_xticks(xaxis)
    #     ax.set_xticklabels(rating_names,rotation=90)
    #     ax.legend()
            
    #     fig.tight_layout() 

    #     plt.show()


    # xaxis = np.arange(14)
    # width = 0.3

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(xaxis - width/2, a_list_predict[i], width, label=unpriv_label_list[i])
    # rects2 = ax.bar(xaxis + width/2, b_list_predict[i], width, label=priv_label_list[i])

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Group Probability after running classifier')
    # ax.set_title('Group Fairness for different Label')
    # ax.set_xticks(xaxis)
    # ax.set_xticklabels(rating_names,rotation=90)
    # ax.legend()
        
    # fig.tight_layout() 

    # plt.show()




def cf_plot(df_predict,df_true,cf_loss_list):

    pred_std, truth_std,pred_mean, truth_mean, pred_prob_mat, truth_prob_mat = find_std_dev(df_predict, df_true)
    
    print(pred_std)
    print('======')
    print(pred_prob_mat)

    print('-----------------------')

    print(truth_std)
    print('======')
    print(truth_prob_mat) 

    pred_mean = pred_mean + 0.00001
    truth_mean = truth_mean + 0.00001
    pred_cv = pred_std / pred_mean
    truth_cv = truth_std / truth_mean


    plt.subplot(1,2,1) 
    plt.plot(range(len(cf_loss_list)),cf_loss_list)
    plt.xlabel('Number of of Epochs')
    plt.ylabel('Counterfactual Loss')

    plt.subplot(1,2,2)
    plt.scatter(truth_cv,pred_cv)
    min_range = min(min(pred_std),min(truth_std))
    max_range = max(max(pred_std),max(truth_std))
    #line =np.linspace(min_range,max_range,100)
    line =np.linspace(0,0.7,100)
    plt.ylabel('CV of the predicted label')
    plt.xlabel('CV of the true label')
    plt.plot(line,line)

    #plt.subplot(2,2,3)



    plt.show()


def scatter_plot_group(df_predict,df_true):

    pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat = find_std_dev(df_predict, df_true)

    for k,label in enumerate(['courageous']):#enumerate(rating_names):

        predicted_probs = pred_prob_mat[[label]].values
        print(predicted_probs)
        true_probs = truth_prob_mat[[label]].values

        num_groups = true_probs.shape[0]
        print(num_groups,"num of groups")
        predicted_probs,true_probs  = predicted_probs.reshape(num_groups,), true_probs.reshape(num_groups,)

        #print(true_probs)
        x_pred,y_pred,x_true,y_true = [], [], [], []

        for i in range(len(predicted_probs)):
            for j in range(i):
                if predicted_probs[i] >= 0.001 and predicted_probs[j] >= 0.001 and true_probs[i] >= 0.001 and true_probs[j] >= 0.001: 
                    x_pred.append(predicted_probs[i])
                    y_pred.append(predicted_probs[j])

                    x_true.append(true_probs[i])
                    y_true.append(true_probs[j])


        #plt.subplot(3,5,k+1)       
        x_pred,y_pred, x_true, y_true = np.array(x_pred),np.array(y_pred),np.array(x_true),np.array(y_true)
        #plt.scatter(np.abs(x_true-y_true),np.abs(x_pred - y_pred),color='blue')
        #plt.scatter(np.abs(),np.abs((x_pred/(y_pred+0.0001))),color = 'red')
        
        plt.subplot(1,2,1)
        true_diff = x_true-y_true
        pred_diff = x_pred - y_pred
        print(true_diff)
        print('--------')
        print(pred_diff)
        xaxis=np.arange(len(x_pred))#np.linspace(0,1,100)
        plt.plot(xaxis,np.zeros(len(x_pred)))
        plt.scatter(xaxis,true_diff,color='blue')
        plt.scatter(xaxis,pred_diff,color='red')
        try:
            for i in range(len(x_pred)):
                plt.arrow(i,true_diff[i],0,pred_diff[i]-true_diff[i],length_includes_head=True,head_width=0.8,head_length=0.05)
        except:
            pass



        
      
        plt.subplot(1,2,2)
        true_ratio = (x_true/(y_true+0.0001))
        pred_ratio = (x_pred/(y_pred+0.0001))
        
        print(true_ratio)
        print('+++++++++')
        print(pred_ratio)
        xaxis=np.arange(len(x_pred))#np.linspace(0,1,100)
        plt.plot(xaxis,np.ones(len(x_pred)))
        plt.scatter(xaxis,true_ratio,color='blue')
        plt.scatter(xaxis,pred_ratio,color='red')
        
        
        try:
            for i in range(len(x_pred)):
                plt.arrow(i,true_ratio[i],0,pred_ratio[i]-true_ratio[i],length_includes_head=True,head_width=0.8,head_length=0.05)
        # plt.xlim(0,1)
        #plt.ylim(0,1)
        except:
            pass
            
        plt.tight_layout()
        plt.savefig('../Plots/'+label+'_predicted.pdf')

        plt.close()
        #plt.show()

def plot_cv(df_predict,df_true):

    pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat = find_std_dev(df_predict, df_true)
    pred_mean = pred_mean + 0.00001
    truth_mean = truth_mean + 0.00001
    pred_cv = pred_std / pred_mean
    truth_cv = truth_std / truth_mean

    plt.scatter(truth_cv,pred_cv)
    min_range = min(min(pred_std),min(truth_std))
    max_range = max(max(pred_std),max(truth_std))
    #line =np.linspace(min_range,max_range,100)
    line =np.linspace(0,0.7,100)
    plt.ylabel('CV of the predicted label')
    plt.xlabel('CV of the true label')
    plt.plot(line,line)

    #plt.subplot(2,2,3)



    plt.show()

#cf_plot(data_frame_predict, data_frame_true,cf_list)
#plot_using_aif(data_frame_predict, data_frame_true)

scatter_plot_group(data_frame_predict, data_frame_true)
#plot_cv(data_frame_predict, data_frame_true)

# plt.xlim(0,40)
# plt.ylim(-0.5,0.7)
# plt.arrow(20,0,0,0.5,length_includes_head=True,head_width=0.8,head_length=0.05)
# plt.show()
