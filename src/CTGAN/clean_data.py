# limpeza.py
import pandas as pd

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Carregar dados do arquivo Parquet"""
        self.data = pd.read_parquet(self.file_path)
    
    def clean_data(self):
        """Limpar e preparar os dados"""
        # Colunas numéricas a serem convertidas para float64
        numeric_cols = ['ListPrice', 'CurrentPrice', 'Cost', 'Quantity', 'StockPieces']
        self.data[numeric_cols] = self.data[numeric_cols].astype('float64')
        
        # Converter a coluna 'Week' para formato datetime
        self.data['Week'] = pd.to_datetime(self.data['Week'])
        
        # Criar identificador único para cada produto
        self.data['ProductID'] = self.data['ModelID'].astype(str) + "_" + self.data['ItemColorId'].astype(str)
        
        # Ordenar os dados por 'Week' e redefinir o índice
        self.data = self.data.sort_values(by='Week').reset_index(drop=True)
        
        # Criar coluna de número da semana
        self.data['Week_Number'] = self.data['Week'].dt.isocalendar().week
        self.data['Week_Number'] = self.data['Week_Number'].astype(int)
        
        # Selecionar colunas finais
        self.data = self.data[['ProductID', 'Week_Number', 'CurrentPrice', 'ListPrice', 'Cost', 'Quantity', 'StockPieces', 'Week']]
        
    def get_clean_data(self):
        """Retorna o DataFrame limpo"""
        return self.data
    
    def save_clean_data(self, output_path):
        """Salva o DataFrame limpo em um arquivo Parquet"""
        self.data.to_parquet(output_path, index=False)

if __name__ == "__main__":
    cleaner = DataCleaner("data/df_complete_cases.parquet")
    cleaner.load_data()
    cleaner.clean_data()
    cleaner.save_clean_data("data/df_cleaned.parquet")