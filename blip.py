import os
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

# Carregar modelo spaCy (melhor para download único)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Baixando o modelo spaCy 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DreamstimeUploadAutomator:
    def __init__(self, pasta_imagens, categorias, num_threads=4):
        self.pasta_imagens = pasta_imagens
        self.categorias = categorias
        self.processor, self.model = self.configurar_modelos()
        self.num_threads = num_threads
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def configurar_modelos(self):
        print("Carregando modelos de análise de imagem...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    def gerar_palavras_chave(self, descricao, max_keywords=20, min_length=2):
        """
        Gera palavras-chave mais longas e relevantes usando spaCy.
        """
        doc = nlp(descricao.lower())
        palavras_chave = set()
        # Adicionar substantivos e adjetivos que não são stop words
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"] and not token.is_stop and len(token.text) >= min_length:
                palavras_chave.add(token.text)
        # Adicionar N-gramas (combinações de 2 palavras)
        for i in range(len(doc)-1):
            if doc[i].pos_ in ["NOUN", "ADJ"] and doc[i+1].pos_ in ["NOUN", "ADJ"]:
                ngram = f"{doc[i].text} {doc[i+1].text}"
                if not any(token.is_stop for token in doc[i:i+2]) and len(ngram) >= (min_length * 2 + 1):
                    palavras_chave.add(ngram)

        return ', '.join(list(palavras_chave)[:max_keywords])

    def analisar_imagem(self, caminho_imagem):
        try:
            imagem = Image.open(caminho_imagem).convert("RGB")
            inputs = self.processor(images=imagem, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(**inputs, max_length=300, num_beams=10)
            descricao = self.processor.decode(output_ids[0], skip_special_tokens=True)
            resumo_titulo = ' '.join(descricao.split()[:12]) + '...' if len(descricao.split()) > 12 else descricao
            palavras_chave = self.gerar_palavras_chave(descricao)
            return {
                'descricao': descricao,
                'resumo_titulo': resumo_titulo,
                'palavras_chave': palavras_chave
            }
        except Exception as e:
            print(f"Erro ao analisar imagem {caminho_imagem}: {e}")
            return {
                'descricao': 'Erro na descrição',
                'resumo_titulo': 'Erro na descrição',
                'palavras_chave': ''
            }

    def processar_imagem_em_lote(self, arquivo_nome):
        caminho_completo = os.path.join(self.pasta_imagens, arquivo_nome)
        return arquivo_nome, self.analisar_imagem(caminho_completo)

    def gerar_arquivo_texto(self):
        nome_arquivo = "dreamstime_upload.txt"
        with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
            
            imagens_a_processar = [arquivo_nome for arquivo_nome in os.listdir(self.pasta_imagens)
                                  if arquivo_nome.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            file_names = []
            image_names = []
            descriptions = []
            keywords_list = []

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                resultados = executor.map(self.processar_imagem_em_lote, imagens_a_processar)

                for arquivo_nome, analise in resultados:
                    file_names.append(arquivo_nome)
                    # Adicionar 'generated' ao final do resumo do título
                    image_names.append(analise['resumo_titulo'] + ' generated')
                    descriptions.append(analise['descricao'])
                    keywords_list.append(analise['palavras_chave'])

            # Imprimir dados formatados em blocos
            arquivo.write(f"Filename: {', '.join(file_names)}\n")
            arquivo.write(f"Image Name: {', '.join(image_names)}\n")
            arquivo.write(f"Descriptions: {', '.join(descriptions)}\n")
             
            # Adicionar categorias
            arquivo.write(f"Category 1: {self.categorias[0] if len(self.categorias) > 0 else ''}\n")
            arquivo.write(f"Category 2: {self.categorias[1] if len(self.categorias) > 1 else ''}\n")
            arquivo.write(f"Category 3: {self.categorias[2] if len(self.categorias) > 2 else ''}\n")

            # Imprimir palavras-chave por imagem
            for i, keywords in enumerate(keywords_list):
               arquivo.write(f"Keywords for {file_names[i]}: {keywords}\n")

            # Adicionar constantes
            arquivo.write("Free: 0\n")
            arquivo.write("W-EL: 1\n")
            arquivo.write("P-EL: 1\n")
            arquivo.write("SR-EL: 1\n")
            arquivo.write("SR-Price: 100\n")

        print(f"Arquivo '{nome_arquivo}' criado com sucesso!")


def solicitar_caminho_imagens():
    while True:
        caminho = input("Digite o caminho completo para a pasta com suas imagens: ").strip()
        if os.path.exists(caminho) and os.path.isdir(caminho):
            return caminho
        else:
            print("Caminho inválido. Por favor, verifique e tente novamente.")

def solicitar_categorias():
    categorias = []
    print("Digite até 3 categorias para suas imagens:")
    for i in range(3):
        categoria = input(f"Categoria {i+1} (deixe em branco para pular): ").strip()
        if categoria:
            categorias.append(categoria)
        else:
            break
    return categorias

def main():
    print("=== Automação de Upload Dreamstime ===")
    # Solicitar caminho das imagens
    pasta_imagens = solicitar_caminho_imagens()
    # Solicitar categorias
    categorias = solicitar_categorias()
    # Solicitar número de threads
    while True:
        try:
            num_threads = int(input("Digite o número de threads para processar as imagens (recomendado 4-8): "))
            if num_threads > 0:
                break
            else:
                print("Por favor, digite um número positivo de threads.")
        except ValueError:
            print("Por favor, digite um número inteiro válido.")
    # Instanciar o automator
    automator = DreamstimeUploadAutomator(pasta_imagens, categorias, num_threads)
    # Gerar arquivo de texto
    automator.gerar_arquivo_texto()

if __name__ == "__main__":
    main()

