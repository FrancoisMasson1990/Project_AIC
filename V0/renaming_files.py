import os

path = "./dataset"
for dir in os.listdir(path):
    try:
        src = os.path.join(path,dir,"chart.csv")
        dst = os.path.join(path,dir,"number_sales.csv")
        os.rename(src,dst)
        src = os.path.join(path,dir,"chart (1).csv")
        dst = os.path.join(path,dir,"sales_usd.csv")
        os.rename(src,dst)
        src = os.path.join(path,dir,"chart (2).csv")
        dst = os.path.join(path,dir,"price_usd.csv")
        os.rename(src,dst)
    except:
        pass