import pickle
import csv

with open('bdm/ctmdata/ctm-b2-d12.pkl','rb') as f2:
    data1 = pickle.load(f2)


with open('bdm/ctmdata/K5-12.csv','r') as f:
    data = csv.reader(f)
    my_dict={row[0]:float(row[1] for row in data}

filename='bdm/ctmdata/ctm-b2-d12b.pkl'
outfile=open(filename,'wb')
pickle.dump(my_dict,outfile)
outfile.close
