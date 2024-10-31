columns_to_work = ['ProductID',
                    'Week_Number', 
                    'CurrentPrice', 
                    'ListPrice', 
                    'Cost', 
                    'Quantity', 
                    'StockPieces', 
                    'Week']

column_types = {
        'ProductID': 'id',
        'ListPrice': 'numerical',
        'CurrentPrice': 'numerical',
        'Cost': 'numerical',
        'Quantity': 'numerical',
        'StockPieces': 'numerical',
        'Week_Number': 'numerical'
    }


cleaned_data_path = "/Pricing_Synthetic/dados/df_cleaned.parquet"

epochs = 50