import numpy as np
from sklearn import tree

def Tree_path(tree,samples):

    '''
    takes tree , pure samples as inputs, could also be not pure samples
    
    returns a list of dictionaries, where key means feature no[0-means-->sample[i][0]], values means condition with thresholds
    one after another

    '''

    number_of_nodes=tree.tree_.node_count
    feature=tree.tree_.feature
    threshold=tree.tree_.threshold
    
    decision_paths=tree.decision_path(samples)
    
    leave_ids=tree.apply(samples)
    
    
    dic=[]
    
    for i in range(0,len(samples),1):
        
        sample_id=i
        
        d=dict()
        
        indexes=decision_paths.indices[decision_paths.indptr[sample_id]:\
                                      decision_paths.indptr[sample_id+1]]
        
        print('sample id: ',sample_id)
        
        comparator=''
        
        for node_id in indexes:
            d[feature[node_id]]=[]
        
        for node_id in indexes:
            
            if leave_ids[sample_id]==node_id:
                
                d.pop(feature[node_id],None)
                
                print(d)
                
                if d not in dic:
                    dic.append(d)
                
                continue

            
            if(samples[sample_id][feature[node_id]] <= threshold[node_id]):
                
                comparator="<="
            else:
                comparator=">"
            
            print("X_test[%s,%s]  %s %s "%(sample_id,feature[node_id],comparator,threshold[node_id]) )
            
            d[feature[node_id]].append(comparator)
            d[feature[node_id]].append(threshold[node_id])
            
    
    print(dic)  
  
    
    return dic

            