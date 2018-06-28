# Hide Text

## Integrantes

Vitor Rossi Speranza - 10262523 e Ygor Franco de Camargo Pontelo - 10295631.

## Área do projeto: Esteganografia

## Descrição

O principal objetivo do nosso trabalho é esconder mensagens em imagens, de modo que estas fiquem muito similares as originais. 
O problema, portanto, é derivado da área de segurança. Note que este trabalho trabalha tanto com a criptografia como a 
esteganografia, visto que o objetivo da primeira é ocultar o significado da mensagem, e a segunda é ocultar a existência desta.

As imagens utilizadas como teste estão disponiveis neste [site](https://www.noticiasaominuto.com/lifestyle/944129/conheca-as-mais-belas-paisagens-naturais-do-brasil).

## Desenvolvimento

Para atingirmos nosso objetivo, primeiramente buscamos um modo de processar a imagem de modo que pudessemos encontrar regiões
da imagens que fossem ruidosas, e que quando sobreescritas, não causassem um impacto significativo na qualidade da imagem
gerada pelo programa. Para isso, aplicamos uma filtragem na imagem, usando uma convolução, que nos reveleva estas regiões.

Após isso, requerimos uma chave de encriptação, que será usada tanto para encriptar o texto como para escrever a mensagem 
na imagem, pedimos também a densidade da imagem, que significa o quanto de espaço o usuário requer para escrever o seu texto 
na imagem, textos maiores requerem uma densidade maior.

O primeiro passo é encriptar o texto passado, para isso usamos uma biblioteca do Python chamada Crypto, e a encriptação usada 
foi a AES, modo CBC. O algoritmo então gera uma sequência aleatória de posições dentro das regiões permitidas através da chave 
dada e escreve nessas posições o texto, guardando o caractere e a próxima posição que ele terá que percorrer, isso será útil no 
momento de recuperar o texto da imagem.

Por fim, mostramos uma comparação entre a imagem gerada e a original, assim como o seu RMSE e as regiões que o algoritmo detectou 
como ruidosas e possíveis para sobreescrever.

Pra recuperar o texto da imagem, basta informar a chave usada e o algoritmo irá percorrer as posições gravadas na imagem, recuperando
o texto, que ainda está encriptado, portanto o algoritmo desencripta o texto e exibe a mensagem ao usuário.

## Resultados

Os resultados que obtivemos foram muito satisfatórios, tendo em vista que as imagens geradas pelo programa são muito próximas
a imagem original, não sendo possível ver diferença notável a olho nú. Os RMSEs das imagens de texte foram baixíssimos, com
valores na média de 0,03. Em questão de tempo de execução o programa se mostra bom, tendo imagens de resoluções altas sendo 
processados em tempo aceitável.
