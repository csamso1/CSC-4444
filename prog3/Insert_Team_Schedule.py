def fixTeam(s):
    
    split = s.split()
    for i,sub in enumerate(split):
        for j,k in enumerate(sub[1:]):
            if k.isupper():
                split.append(sub[:j+1])
                split.append(sub[j+1:])
                del split[i]
                break
    split = list(set(split))
    split = sorted(split, key=len)
    split = [s for s in split if len(s)>1]
    if s.startswith("Det") or s.startswith("Clev") or s.startswith("Wash") or s.startswith("Balt") or s.startswith("Jack") or s.startswith("Cin") or s.startswith("Indi") or s.startswith("Minn") or s.startswith("Chi") or s.startswith("Hou") or s.startswith("Buff") or s.startswith("Pitt") or s.startswith("Ten") or s.startswith("Phil") or s.startswith("Car"):
        split = split[::-1]
    if len(split)>2 and (s.startswith("Tamp") or s.startswith("Kan") or s.startswith("Gre")):
        split = [split[1],split[0],split[2]]
    if len(split)>2 and (s.startswith("San F") or s.startswith("St. L") or s.startswith("New O")):
        split = [split[0],split[2],split[1]]
    return " ".join(split)

def main():
    datadir = "./NFL_Data/" 
    filename = "2010DefData.csv"

    import pandas as pd
    import sys
    sys.stdout = open('Insert_2010_Team_Schedule.sql', 'w')

    df = pd.read_csv(datadir + filename)
    df = df.drop('Unnamed: 0', 1)
    attr_list = list(df.columns)
    table_name = "K_Stats"

    print("INSERT INTO  "+table_name)

    for i,j in enumerate(attr_list):
         attr_list[i]= j.replace(" ","_")
    print("("+str(attr_list).replace("[","").replace("\'","").replace("]","")+")")

    print("VALUES")

    for n,row in enumerate(df.values):
        print("(",end="")
        for i,col in enumerate(df.iloc()[n]):
            end = ", " if i<len(attr_list)-1 else ")"
            if attr_list[i]=="Team" and i==11:
                col = fixTeam(col)
            if col.__class__ == str:
                col = "\'"+col+"\'"
            print(col,end=end)
        print("," if n<len(df.values)-1 else "")
        
    print(";")

if __name__ == "__main__":main()