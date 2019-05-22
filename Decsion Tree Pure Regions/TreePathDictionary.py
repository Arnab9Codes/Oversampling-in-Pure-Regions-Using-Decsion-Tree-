import numpy as np


def sample_generation_dictionary(dic,X_train):


    '''
	inputs:

	take dictionary generated from Tree_path function
	and X_train

	ouputs:

	simplified dictionary list for sample generation

    '''
    
    r,c=X_train.shape
    
    min_max_dic=dict()
    
    for i in range(0,c,1):
        min_max_dic[i]=[]
        
        mn=X_train[:,i].min()
        mx=X_train[:,i].max()
        
        min_max_dic[i].append(mn)
        min_max_dic[i].append(mx)

    
    #print("min max dictionary: ",min_max_dic)
    
    gen_sample=[]
    
    gen_dic_list=[]#simplified dictionary generated from dic for sample generation

	'''
	Each dictionary in gen_dic_list represents a dicitonary.
	for each key representing a list of values

	1st value means (less than equal to--->value)
	2nd value means (greater than--------->value)

	_  means no value of that respective position,
	do not consider that posision.

	[5,6]--> means less than equal to 5, greater than 6
	[_,5.5]-->means only greater than 5.5 condition exists, no less that equal to exists
	[5.5,_]-->means the inverse of previous statement

	'''
    
    for i in range(0,len(dic),1):
        
        gen_dic=dict()
        
        for k, v in dic[i].items():
            gen_dic[k]=[]
        
        gen_dic_list.append(gen_dic)
            
    
    for i in range(0,len(dic),1):
        
        mn_list=[]
        mx_list=[]
        
        for k,v in dic[i].items():
            
            all_list=dic[i][k]
            
            for j in range(0,len(all_list),2):
                if all_list[j]=='<=':
                    mn_list.append(all_list[j+1])
                else:
                    mx_list.append(all_list[j+1])
            if len(mn_list)!=0:
                gen_dic_list[i][k].append(min(mn_list))#less than equal to smallest threshold
            else:
                gen_dic_list[i][k].append('_')
                
            if len(mx_list)!=0:
                gen_dic_list[i][k].append(max(mx_list))#greater than to largest threshold
            else:
                gen_dic_list[i][k].append('_')
            
            mn_list.clear()
            mx_list.clear()
    
    
    #print("sample generation dictionnary: \n")
    #for i in range(len(gen_dic_list)):
        #print(gen_dic_list[i])





    
    return gen_dic_list



    




