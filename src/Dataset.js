if(!window.net){
    window.net = {};
}

net.data = {};

/**
* A Dataset Sample

* @param {Array} features 
* @param {Array} desired_values 

* @returns {Object}
*/
net.data.Sample = function( features, desired_values ){
    //Validations
    if( !features ){
        throw Error(`features is madatory!`);
    }
    if( !desired_values ){
        throw Error(`desired_values is madatory!`);
    }
    if( !(features instanceof Array && desired_values instanceof Array) ){
        throw Error(`The "features" and "desired_values" must be Arrays!`);
    }

    let context = {};

    context.features       = features;
    context.desired_values = desired_values;

    context.getFeatures = function(){
        return context.features;
    }

    context.getDesiredValues = function(){
        return context.desired_values;
    }

    /**
    * Get a feature and they desired value
    */
    context.getPosition = function(index){
        if( index == undefined ){
            throw Error(`Need a index!`);
        }
        if(index < 0){
            throw Error(`Dont negatives!`);
        }
        if( index > context.features.length ){
            throw Error('Exceded the length!');
        }

        return [ context.features[index], context.desired_values[index] ];
    }

    context.getFeature = function(index){
        return context.features[index];
    }

    context.getDesiredValue = function(index){
        return context.desired_values[index];
    }

    return context;
}

/**
* A Dataset for store Samples
*  
* @param {Array} my_dataset_structure
*
* @returns {Object}
*/
net.data.Dataset = function( my_dataset_structure ){
    //Validations
    if( !my_dataset_structure ){
        throw Error(`my_dataset_structure is madatory!`);
    }
    if( !(my_dataset_structure instanceof Array) ){
        throw Error(`my_dataset_structure must be Array!`);
    }

    let context = {};

    context.samples = [];
    context._my_dataset_structure = my_dataset_structure;

    /**
    * Convert my_dataset_structure to Objects to make easy manipulation
    */
    context._my_dataset_structure.forEach(function(my_sample){
        let sample_features       = my_sample[0];
        let sample_desired_values = my_sample[1];

        //Add the Sample to the context.samples array
        context.samples.push( 

            new net.data.Sample( sample_features, 
                                 sample_desired_values )

        );
    });

    context.getSamples = function(){
        return context.samples;
    }

    context.getSample = function(index){
        if( index == undefined ){
            throw Error(`Need a index!`);
        }
        if(index < 0){
            throw Error(`Dont negatives!`);
        }
        if( index > context.samples.length ){
            throw Error('Exceded the length!');
        }

        return context.samples[index];
    }

    return context;
}