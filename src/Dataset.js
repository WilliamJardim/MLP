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
        if( index == undefined ){
            throw Error(`Need a index!`);
        }
        if(index < 0){
            throw Error(`Dont negatives!`);
        }
        if( index > context.features.length ){
            throw Error('Exceded the length!');
        }

        return context.features[index];
    }

    context.getDesiredValue = function(index){
        if( index == undefined ){
            throw Error(`Need a index!`);
        }
        if(index < 0){
            throw Error(`Dont negatives!`);
        }
        if( index > context.features.length ){
            throw Error('Exceded the length!');
        }
        
        return context.desired_values[index];
    }

    context.getDataset = function(){
        return context._dataset;
    }

    context.getIndex = function(){
        return context._index;
    }

    return new Proxy(context, {
        get: function(target, prop, receiver) {

          if (typeof prop === 'string' && !isNaN(prop) ) {

              if( Number(prop) === 0 ){
                 return target.features;

              }else if( Number(prop) === 1 ){
                 return target.desired_values;

              }else{
                 throw Error(`Invalid prop index!`);
              }
    
          }
          
          return Reflect.get(target, prop, receiver);
        },

        set: function(target, prop, value) {
          return Reflect.set(target, prop, value);
        }
    });
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
    context.flags = ['dataset'];
    context.internalType = 'dataset';

    context.samples = [];
    context._my_dataset_structure = my_dataset_structure;

    /**
    * Convert my_dataset_structure to Objects to make easy manipulation
    */
    context._my_dataset_structure.forEach(function(my_sample, my_sample_index){
        let sample_features       = my_sample[0];
        let sample_desired_values = my_sample[1];

        //Add the Sample to the context.samples array
        context.samples[my_sample_index] = new net.data.Sample( sample_features, 
                                                                sample_desired_values );

        context.samples[my_sample_index]._index   = my_sample_index;             
        context.samples[my_sample_index]._dataset = context;    
        context.samples[my_sample_index].params   = {};
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

    context.copyWithin = function(){
        return [... context.samples.copyWithin()];
    }

    context.forEach = function(forFn){
        return context.samples.forEach(forFn);
    }

    return new Proxy(context, {
        get: function(target, prop, receiver) {

          if (typeof prop === 'string' && !isNaN(prop)) {
            return target.getSample( Number(prop) );

          }else if (typeof prop === 'number' && !isNaN(prop)) {
            return target.getSample( prop );
          }

          return Reflect.get(target, prop, receiver);
        },

        set: function(target, prop, value) {
          return Reflect.set(target, prop, value);
        }
    });
}