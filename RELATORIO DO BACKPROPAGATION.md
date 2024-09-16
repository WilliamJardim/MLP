# CAMADA DE SAIDA
Primeiro de tudo, ele chama o método "calculate_derivatives_of_output_units" que vai calcular as derivadas das unidades da camada de saida(a ultima camada da rede)
pra isso essa função recebe os parametros:  output_estimated_values, 
                                            desiredOutputs, 
                                            list_to_store_gradients_of_units, 
                                            list_to_store_gradients_for_weights 

o parametro "output_estimated_values" contém as estimativas(saidas das unidades da camada de saida, obtidas no feedforward)

O parametro "desiredOutputs" contém as saidas desejadas para cada unidade de camada de sáida

os parametros "list_to_store_gradients_of_units" e "list_to_store_gradients_for_weights" também são passados para essa função "calculate_derivatives_of_output_units" 

Com isso, a função percorre todas as unidades da camada de saida,
usando um forEach: "context.getOutputLayer().getUnits().forEach(function( output_unit, 
                                                                          output_unit_index ) "

Então, para cada unidade da camada de saida, ele calcula a diferença entre o valor estimado da unidade e o valor desejado para a unidade,
depois disso, ele multiplica essa diferença pela derivada da função da unidade, obtendo assim a derivada em relação a unidade

Ele salva essa derivada na variavel "list_to_store_gradients_of_units" passada como parametro
criando uma chave layer1 dentro dela:
{
    "layer1": {
        "unit0": -0.029379162360871295,
        "unit1": 0.13553299084477086
    }
}


Após isso, ele dispara outro forEach percorrendo cada peso da unidade, para calcular as derivadas em relação a cada peso da unidade
De forma similar a estrutura anterior:
{
    "layer1": {
        "unit0": [
            -0.0283329649798022,
            -0.026985521432844786
        ],
        "unit1": [
            0.13070663608596025,
            0.12449056186061569,
            null
        ]
    }
}

porém, aqui ele calculou para cada peso

Com isso, ele calcula as derivadas de cada unidade da camada de saida, em relação a cada parametro delas

ele começou pela layer1 por que primeiro, antes de começar a retropropagar o erro, nós precisamos calcular a derivada as unidades da camada de saida, em relação cada parametro dessas unidades.

COM ISSO ESSA PRIMEIRA ETAPA DO BACKPROPAGATION FOI CONCLUIDA!

Agora, a partir dessas derivadas da camada de saida, nós vamos conseguir calcular as derivadas das camadas anteriores(as camadas ocultas):

# CAMADAS OCULTAS
Feito isso, o próximo passo é calcular as derivadas das camadas ocultas.

Ele cria uma variavel chamada "currentLayerIndex = number_of_layers-1-1", ou seja essa variavel é uma variavel controladora, que começa sendo igual a quantidade de camadas - 1 - 1, pois, queremos ignorar a camada de entrada, e queremos ignorar tambem a camada de saida(que já calculamos acima)

Ai apartir da linha 940, ele cria um loop while:  

        /**
        * While the "while loop" not arrived the first hidden layer
        * The first hidden layer(that have index 0, will be the last layer that will be computed) 
        */
        while( currentLayerIndex >= 0 )
        {
            ... corpo do while
        }

Esse while faz o seguinte: **ENQUANTO NÂO CHEGAR NA PRIMEIRA CAMADA OCULTA, EXECUTA OS CÓDIGOS DENTRO DELE**

Como podemos ver, esse while começa iterando sobre a ultima camada oculta, e vai decrementando/retrocedendo para tráz, até chegar na primeira camada oculta

O código dentro desse while é bem simples

Ele obtem os dados da camada atual da iteração do while:

            //Current layer data
            let current_layer          = context.getLayer( currentLayerIndex );
            let current_layer_inputs   = inputs_of_each_layer[ `layer${ currentLayerIndex }` ];
            let current_layer_outputs  = outputs_of_each_layer[ `layer${ currentLayerIndex }` ];

Nessa primeira iteração desse loop while, os valores são os seguintes:
    currentLayerIndex = 0;
    current_layer_inputs = [2.7810836, 2.550537003]
    current_layer_outputs = {unit0: 0.9643898158763548, unit1: 0.9185258960543243};

pois, ele começa na ultima camada oculta, e estamos na primeira iteração.

Ele manipula as variaveis "list_to_store_gradients_of_units" e "list_to_store_gradients_for_weights" criadas, que são objetos. Ele manipula criando chaves para a camada atual da iteração atual

Portando, agora o objeto "list_to_store_gradients_of_units" possui uma chave para a layer0

{
    "layer1": {
        "unit0": -0.029379162360871295,
        "unit1": 0.13553299084477086
    },
    "layer0": {}
}

Após isso, ele obtem os dados da próxima camada(a camada subsequente, ou seja, a camada que vem depois da camada atual da iteração do loop while)
E armazena essas informações em variaveis, Assim:

            /**
            * Next layer data:
            */
            let next_layer_index               = currentLayerIndex + 1;
            let next_layer                     = context.getLayer( next_layer_index );

O objeto "next_layer" guarda as informações da camada seguinte.
e a variavel "next_layer_index" contém o indice da camada seguinte.

Após isso, ele acessa os gradientes da camada seguinte(usando o valor da variavel "next_layer_index"), que foram calculados na etapa anterior
E armazena essas informações em variaveis, Assim:

            /**
            * Get the gradients(of the units) of the next layer
            * These gradients will be used in lines below:
            */
            let next_layer_gradients           = list_to_store_gradients_of_units[ `layer${ next_layer_index }` ];

Nesse caso, os gradientes das unidades da camada seguinte foram:

 {
    "unit0": -0.029379162360871295,
    "unit1": 0.13553299084477086
 }

NOTE: Estamos usando as derivadas em relação a cada unidade da camada seguinte

Após isso, ele começa a percorrer todas as unidades da camada atual, usando um forEach
Assim:

            /**
            * For each unit in CURRENT HIDDEN LAYER
            */
            current_layer.getUnits().forEach(function( current_hidden_layer_unit, 
                                                       the_unit_index)


Dentro desse forEach, ele cria a variavel "let current_unit_inputs   = [... current_layer_inputs.copyWithin()];" que vai armazenar as entradas das unidades da camada atual, que são as entradas da camada atual(pois todas as unidades de uma camada recebem exatamente as mesmas entradas(Isto é, as entradas dessa camada em questao) ). Isso logo será usado para calcular as derivadas em relação a cada peso da unidade

Com isso, na linha a seguir, vamos calcular a derivada da unidade atual do forEach

                /**
                * Calculate the derivative of the current unit UH
                * Relembering that, 
                * The derivative of a unit in a hidden layer will always depend of the derivatives of the next layer
                */ 
                let current_hidden_unit_LOSS  = context.calculate_hidden_unit_derivative( 
                                                                                          current_hidden_unit_index    = hidden_unit_index, 
                                                                                          next_layer_units             = next_layer.getUnits(),
                                                                                          next_layer_units_gradients   = next_layer_gradients 
                                                                                        );

(CONTINUAR ....)


