from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd


class PricingCTGAN:
    def __init__(self, data_path):
        self.data_path = data_path
        self.metadata = None
        self.synthesizer = None
        self.data = None

    def load_data(self):
        """Carregar dados limpos"""
        self.data = pd.read_parquet(self.data_path)
    
    def setup_metadata(self, table_name="ecommerce-products", column_types=None):
        """
        Configurar o metadata para os dados.

        Parameters:
        - table_name (str): Nome da tabela para o metadata.
        - column_types (dict): Dicionário onde as chaves são nomes das colunas e os valores são os tipos ('id', 'numerical', 'categorical', etc.).
        """
        # Detectar metadata automaticamente
        self.metadata = Metadata.detect_from_dataframe(data=self.data, table_name=table_name)
        
        # Atualizar as colunas no metadata com os tipos fornecidos pelo usuário
        if column_types:
            for column_name, sdtype in column_types.items():
                self.metadata.update_column(column_name=column_name, sdtype=sdtype)
        
    def train_synthesizer(self, epochs=6000, batch_size=64, cuda=True, enforce_rounding=False, verbose=True):
        """
        Treinar o modelo CTGAN com parâmetros ajustáveis.

        Parameters:
        - epochs (int): Número de épocas para o treinamento.
        - batch_size (int): Tamanho do batch para o treinamento.
        - cuda (bool): Se True, utiliza GPU.
        - enforce_rounding (bool): Se True, força arredondamento nos valores gerados.
        - verbose (bool): Se True, exibe o progresso do treinamento.
        """
        self.synthesizer = CTGANSynthesizer(
            metadata=self.metadata,
            enforce_rounding=enforce_rounding,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            verbose=verbose
        )
        self.synthesizer.fit(self.data)
        
    def save_synthesizer(self, output_path):
        """Salvar o modelo treinado"""
        self.synthesizer.save(output_path)
    
    def visualize_metadata(self):
        """Visualizar o metadata configurado"""
        self.metadata.visualize()