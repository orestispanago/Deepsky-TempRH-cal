import mysql.connector
import pandas as pd

def select_station(station):
    mydb = mysql.connector.connect(
        host="DatabaseIP",
        port="3360",
        user="ReadOnlyUser",
        passwd="ReadOnlyPassword",
        database="deepsky"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM measurements WHERE stationID=%s",(station,))
    data = pd.DataFrame(mycursor.fetchall())
    data.columns = mycursor.column_names
    data.set_index("time", inplace=True)
    mydb.close()
    return data

for i in range(1,7):
    df = select_station(i)
    df.to_csv(f"raw/station{i}.csv")