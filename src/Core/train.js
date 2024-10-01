/**
* Model training loop 
* 
* @param {Array} train_samples
* @param {Number} number_of_epochs
*/
net.MLP.prototype.train = function( train_samples, 
                                    number_of_epochs 
){
    let context = this; //The model context

    //Validations
    //If is a array or a instance of net.data.Dataset, is acceptable and compative type
    if( !(train_samples instanceof Array || train_samples.internalType == 'dataset' ) ){
        throw Error(`The train_samples=[${train_samples}] need be a Array instance!`);
    }

    if( train_samples.length == 0 ){
        throw Error(`The train_samples is a empty Array!`);
    }

    if( number_of_epochs == undefined || number_of_epochs == null ){
        throw Error(`The number_of_epochs is undefined!`);
    }

    if( isNaN(number_of_epochs) ){
        throw Error(`The number_of_epochs is NaN!`);
    }

    if( number_of_epochs == Infinity ){
        throw Error(`The number_of_epochs never be inifinity!`);
    }

    if( String(number_of_epochs).indexOf('.') == true ){
        throw Error(`The number_of_epochs=${number_of_epochs} is a invalid value!. The value ${number_of_epochs} should be Integer!`);
    }

    //Task validation
    if( context.task == 'classification' || context.task == 'logistic_regression' || context.task == 'binary_classification'){

        //Check if some disired value of the samples ARE NOT BINARY
        let someDesiredValueIsNotBinary = [... train_samples.copyWithin()].some( 
        function(entry) {
            return Object.values(entry[1]).every(function(value) {
                return (value != 1 && value != 0 && typeof value != 'boolean')  
            })
        })

        if( someDesiredValueIsNotBinary == true ){
            throw Error(`dataset problem!. Some desired values are not binar!. But, In task of ${context.task}, Should be only 0 and 1, or boolean`);
        }

    }

    //Sample validation
    let secure_copy_of_samples_for_validations = [... train_samples.copyWithin()];
    validations.throwErrorIfSomeSampleHaveObjectsArraysInsteadValues( secure_copy_of_samples_for_validations );
    validations.throwErrorIfSomeSampleAreStringsOrCharacters( secure_copy_of_samples_for_validations );
    validations.throwErrorIfSomeSampleAreIncorrectArrayLength( secure_copy_of_samples_for_validations );
    validations.throwErrorIfSomeSampleAreDiffentLengthOfInputsThatTheInputLayer( context.input_layer.inputs , secure_copy_of_samples_for_validations );

    //Start train
    let training_result = {};

    switch( context.getTrainingType() ){
        case 'online':
            training_result = context.online_train(train_samples, number_of_epochs);
            break;

        case 'batch':
        case 'fullbatch':
            training_result = context.fullbatch_train(train_samples, number_of_epochs);
            break;

        case 'minibatch':
            training_result = context.minibatch_train(train_samples, number_of_epochs);
            break;

        default:
            throw Error(`Invalid training type ${ context.getTrainingType() }!`);
    }

    return {
        model: context,
        last_total_loss: training_result.last_total_loss,
        loss_history: training_result.loss_history,
        initial_loss: training_result.loss_history[0],
        after_first_epoch_loss: training_result.loss_history[1],
        final_loss: training_result.loss_history[training_result.loss_history.length-1],
    };

}