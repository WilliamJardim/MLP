Uma época completa, passos lógicos seguidos:

Voce tem as entradas da camada de entrada

Com essas entradas, Faz o feedforward( sample_inputs )

e o parametro sample_inputs da função feedforward possui 2 entradas(ou seja é um Array de 2 elementos)

Ele inicializa as variáveis "inputs_of_each_layer" e "outputs_of_each_layer" como objetos {}, e atrela isso ao modelo(contexto pai) usando o context.vinculate em ambas

Depois disso, ele logo abaixo na linha 622 chama a função 

"context.get_first_hidden_layer().setInputs( [... sample_inputs] );"

O método setInputs manipulou o objeto "inputs_of_each_layer"(que foi vinculado com o modelo por meio da função vinculate), gravando nele as entradas que essa primeira camada oculta recebeu(que nesse caso foram as entradas da camada de entrada)
E como resultado, o objeto "inputs_of_each_layer" que antes era {}, ficou: "{ layer0: [2.7810836, 2.550537003] }" 
ou seja, o método setInputs criou a chave "layer0" que corresponde a primeira camada oculta(cujo indice é 0), contendo o array [2.7810836, 2.550537003] que é justamente as entradas da camada de entrada

e tambem, o método setInputs no final vincula essa informação tambem na camada na propriedade LAYER_INPUTS em "this_layer_ref[ 'LAYER_INPUTS' ]". Isso faz com que chamar o método <camada>.getInputs() retorne as entradas dessa camada

Ou seja, o "context.get_first_hidden_layer()" retorna o objeto da primeira camada oculta da rede

Cria variavel "final_outputs" como um Array vazio [] que no final de tudo vai contar as saidas finais(estimativas da camada de saida)

Depois disso, ele começa a percorrer todas as camadas da rede deis da primeira até a ultima
Usando o "context.getLayers().forEach(function( current_layer, layer_index) {....corpo da função} "
o "context.getLayers()" retorna um Array que contém os objetos das camadas da rede.
o layer "layer_index" na primeira iteração desse forEach começa sendo 0, pois ele está lendo a primeira camada oculta(cujo indice é 0), 

Ai ele sem seguida nesse loop, obtem as saidas(estimativas) das unidades dessa primeira camada oculta, usando a função "let units_outputs = current_layer.get_Output_of_Units();" 
Esse método faz com que cada unidade da camada atual receba as mesmas entradas da camada atual, e produzam suas saidas independentes(isso é, cada unidade produz a sua própia saida, após todas essas unidades processarem as mesmas entradas). 

Ai nas linhas 645 a 678 ele acessa o objeto "outputs_of_each_layer", criando a chave layer0, e mapeia as saidas dessas unidades da primeira camada da rede dentro dessa chave layer0, pra ficar organizado.
Então, o resultado atual do objeto "outputs_of_each_layer" ficou assim "{ layer0: {unit0: 0.9643898158763548, unit1: 0.9185258960543243} }"

Ou seja, dentro da camada layer0, ele tem suas unidades, sendo elas "unit0" e "unit1", e cada uma tem sua respectiva saida(estimativa)

Ai, com essas saidas calculadas e estruturadas,
ele faz uma checagem IF

 "current_layer.notIs('output')", isso é, ele pergunta "a camada atual ela é a camada de saida?"
se o resultado dessa condição for true(e nesse caso vai ser, por que a primeira camada cujo indice é 0 não é a camada de saida), então, ele entra no bloco desse IF

Com isso, dentro desse bloco do IF, ele na linha 659 obtem o objeto da proxima camada(ou seja, a camada seguinte, ou seja, a camada que vem depois da camada atual), e salva esse objeto na variavel "next_layer"

E logo apos obter o objeto da "next_layer", dentro do contexto da proxima camada, ele chama aquele método "setInputs" novamente, porém passando as saidas da camada atual(isso é, da primeira camada oculta cujo indice é 0, da iteração atual do forEach do getLayers), 
atrávez do "next_layer.setInputs( units_outputs );" na linha 664
Ou seja, ele vai criar a chave para a proxima camada "layer1"

E o resultado disso no objeto "inputs_of_each_layer" foi o seguinte:

{
    "layer0": [
        2.7810836,
        2.550537003
    ],
    "layer1": [
        0.9643898158763548,
        0.9185258960543243
    ]
}

Após isso, ele encerra essa iteração, e vai pra segunda iteração desse forEach,
então, ele começa denovo

Agora novamente na linha 640, com o parametro "layer_index" valendo 1(ou seja, agora nessa segunda iteração estamos na segunda camada da rede)
ele obtem novamente as saidas das unidades dessa segunda camada usando o código "let units_outputs = current_layer.get_Output_of_Units();"

Agora, para produzir as saidas dessas unidades da segunda camada, o método "get_Output_of_Units" usou as entradas referentes a essa segunda camada,
que o passo anterior registrou no objeto "inputs_of_each_layer" como layer1: ARRAY
Ou seja, cada unidade da segunda camada recebeu as saidas da primeira camada oculta

Com essas saidas calculadas, ele vai armazenar no objeto "outputs_of_each_layer"
na chave layer1
E agora o objeto "outputs_of_each_layer" ficou assim:
{
    "layer0": {
        "unit0": 0.9643898158763548,
        "unit1": 0.9185258960543243
    },
    "layer1": {
        "unit0": 0.8094918973879515,
        "unit1": 0.7734292563511262
    }
}

ou seja, a segunda camada oculta(nomeada layer1, pois o indice dela é 1), contém duas unidades: sendo elas "unit0" e "unit1", que produziram as saidas(contidas no objeto acima)

ou seja isso ficou registrado tambem assim como no passo anterior!

Depois ele vai fazer aquela condição IF
"current_layer.notIs('output')"
que agora vai retornar false, visto que a minha rede neural desse exemplo possui apenas 2 camadas(a primeira camada de indice 0 e a segunda camada de indice 1). No caso essa segunda camada é a camada de saida, pois isso a condição retornou false

então, ele não vai chamar o getNextLayer e nem o setInputs

em vez disso, ele vai cair no bloco de outro IF
 current_layer.is('output')
que vai retornar true, e portanto irá executar o bloco "final_outputs = units_outputs;" que irá conter as estimativas(saidas finais da propagação da rede neural)

(FIM DA EXECUÇÂO DO FEEDFORWARD)

ai no backpropate_sample,
após executar o feedforward descrito acima

ele vai separa os resultados dele em 3 variaveis:

        let output_estimated_values  = feedforward_data.getEstimatedOutputs();
        let inputs_of_each_layer     = feedforward_data.getInputsOfEachLayer();
        let outputs_of_each_layer    = feedforward_data.getOutputsOfEachLayer();

E vai em seguida, criar dois objetos

"list_to_store_gradients_of_units" e "list_to_store_gradients_for_weights"
que vão ser responsáveis por armazenar os gradientes de camada unidade de cada camada

(CONTINUAR....)















