# Implementação

A implementação até o momento consiste em encriptografar um texto, usando a encriptação AES com o padrão CBC, 
e através da chave passada para a criptografia, selecionar regiões aleátorias da imagem, nas quais estejam
aptas para serem escritas, e escrever o texto nessas regiões. A análise dessas regiões foi feita por meio de 
uma filtragem inicial da imagem, que mostrará as partes mais ruidosas da imagem, e assim, onde podemos escrever
sem causar uma distorção significativa na imagem resultante.

# Pŕoxima etapa

O que será implementado futuramente será a comparação da imagem resultante com a imagem original, assim como quais
as regiões que foram sobrescritas, mostrando assim, a eficiência do algoritmo e o RMSE.

*Um teste já está no repositório, assim como o código com o processo atual*.
