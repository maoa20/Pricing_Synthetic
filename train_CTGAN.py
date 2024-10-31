from src.CTGAN import PricingCTGAN
import os
from src.config import *

DIR_HOME = os.environ['HOME']

def main():
    trainer = PricingCTGAN(DIR_HOME + cleaned_data_path)
    trainer.load_data()
    trainer.setup_metadata(table_name="ecommerce-products", column_types=column_types)
    trainer.visualize_metadata() 

    trainer.train_synthesizer(
        epochs=1,
        cuda=True,
        enforce_rounding=True,
        verbose=True
    )

    model_path = f"{DIR_HOME}'/Pricing_Synthetic/models/ctgan_synthesizer.pkl"
    trainer.save_synthesizer(model_path)
    print(f"Modelo CTGAN treinado salvo em: {model_path}")

if __name__ == "__main__":
    main()

