aime <-
function(data.in, data.out, confounder=NULL, do.normalize=TRUE, test.proportion= 0.2, ncomp=4, max.dropout=0.4, min.dropout=0.05, flat.dropout=FALSE, in.layers=NA, out.layers=NA,layer.shrink.factor=2, encoder.layer.sizes=NA, decoder.layer.sizes=NA, activation="relu", max.epochs=100, importance.permutations=1, pairwise.importance=FALSE)
{
   
	if(!is.null(confounder)) if(is.null(nrow(confounder))) confounder<-matrix(confounder, ncol=1)
	temp.path<-tempdir()
	
	if(do.normalize)
	{
		for(i in 1:ncol(data.in)) data.in[,i]<-(data.in[,i]-mean(data.in[,i]))/sd(data.in[,i])
		for(i in 1:ncol(data.out)) data.out[,i]<-(data.out[,i]-mean(data.out[,i]))/sd(data.out[,i])
		if(!is.null(confounder))
		{
			for(i in 1:ncol(confounder)) confounder[,i]<-(confounder[,i]-mean(confounder[,i]))/sd(confounder[,i])
		}
	}
	
    test.sel<-sample(nrow(data.in), round(nrow(data.in)*test.proportion))
    train.in<-data.in[-test.sel,]
    train.out<-data.out[-test.sel,]
    test.in<-data.in[test.sel,]
    test.out<-data.out[test.sel,]
	train.confounder<-confounder[-test.sel,]
	test.confounder<-confounder[test.sel,]

    
    if(is.na(encoder.layer.sizes[1]))
    { 
        if(is.na(in.layers))
        {
            encoder.layer.sizes<-layer.shrink.factor^(-1:-100)*ncol(train.in)
            encoder.layer.sizes<-round(encoder.layer.sizes[encoder.layer.sizes >= 2*ncomp])
        }else{
            encoder.layer.sizes<-round(exp(seq(log(ncol(train.in)), log(ncomp), length.out=in.layers+2)))
            encoder.layer.sizes<-encoder.layer.sizes[c(-1, -length(encoder.layer.sizes))]
        }
    }
    if(is.na(decoder.layer.sizes[1])) 
    {
        if(is.na(out.layers))
        {
            decoder.layer.sizes<-layer.shrink.factor^(-100:-1)*ncol(train.out)
            decoder.layer.sizes<-round(decoder.layer.sizes[decoder.layer.sizes >= 2*ncomp])
        }else{
            decoder.layer.sizes<-round(exp(seq(log(ncomp), log(ncol(train.out)), length.out=out.layers+2)))
            decoder.layer.sizes<-decoder.layer.sizes[c(-1, -length(decoder.layer.sizes))]
        }
    }

    if(flat.dropout)
    {
       in.dropouts<-rep(max.dropout, length(encoder.layer.sizes)+1)
       out.dropouts<-rep(max.dropout, length(decoder.layer.sizes)+1)

    }else{
	   in.dropouts<-exp(seq(log(max.dropout), log(min.dropout), length.out=length(encoder.layer.sizes)+1))
	   out.dropouts<-exp(seq(log(min.dropout), log(max.dropout), length.out=length(decoder.layer.sizes)+1))
	}

    #### structure

    k_clear_session()
    my.encoder<-my.decoder<-NULL
	
	in.vars<-ncol(train.in)
    input_layer <- layer_input(shape = c(ncol(train.in)))


    encoder.code.file<-paste(temp.path,"/encoder_code.txt",sep="")
    fileConn<-file(encoder.code.file)
    
    
    to.write<-c(
    "my.encoder <-",
    "input_layer %>%")
    
    for(i in 1:length(encoder.layer.sizes))
    {
        to.write<-c(to.write, paste("layer_dropout(rate =", in.dropouts[i], ") %>%"))
        if(i != length(encoder.layer.sizes))
		{
			to.write<-c(to.write, paste("layer_dense(units =", encoder.layer.sizes[i], ", activation = \"",activation, "\") %>% ", sep=""))
		}else{
			to.write<-c(to.write, paste("layer_dense(units =", encoder.layer.sizes[i], ")", sep=""))
		}
     }
	#to.write<-c(to.write, paste("layer_dropout(rate =", in.dropouts[i+1], ") %>%"))
    #to.write<-c(to.write, paste("layer_dense(units = ", ncomp, ")"))
    
    
    write(to.write, fileConn)
    close(fileConn)
    
    #message("encoder file written")
    

    source(encoder.code.file, local=TRUE)
    
	encoder.out<-my.encoder %>% 
					layer_dropout(rate = min.dropout ) %>%
					layer_dense(units =  ncomp )
					
    #message("encoder file loaded")
    ###
    
    decoder.code.file<-paste(temp.path,"/decoder_code.txt",sep="")
    fileConn<-file(decoder.code.file)
    



	if(is.null(confounder))
	{
		to.write<-c(
				"my.decoder <-",
				"encoder.out %>%")
    }else{
	    auxiliary_input <- layer_input(shape = c(ncol(confounder)), name = 'aux_input')

		to.write<-c(
				"my.decoder <-",
				"layer_concatenate(c(encoder.out, auxiliary_input)) %>%")
	}
	
    for(i in 1:length(decoder.layer.sizes))
    {
		to.write<-c(to.write, paste("layer_dropout(rate =", out.dropouts[i], ") %>%"))
        to.write<-c(to.write, paste("layer_dense(units =", decoder.layer.sizes[i], ", activation = \"",activation, "\") %>% ", sep=""))

    }

    to.write<-c(to.write, paste("layer_dropout(rate =", out.dropouts[i+1], ") %>%"))
    to.write<-c(to.write, paste("layer_dense(units =", ncol(train.out), ")"))
    
    write(to.write, fileConn)
    close(fileConn)
    
    source(decoder.code.file, local=TRUE)
    
    #message("encoder and decoder files loaded")
    
	if(is.null(confounder))
	{
		autoencoder_model <- keras_model(inputs = input_layer, outputs = my.decoder)
    }else{
		autoencoder_model <- keras_model(c(input_layer,auxiliary_input), outputs = my.decoder)
	}
	
    autoencoder_model %>% compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics = c('accuracy')
    )
    
    summary(autoencoder_model)
    #####
	
	b<-new("list")
    
	if(is.null(confounder))
	{
		history <-
		autoencoder_model %>%
		keras::fit(train.in,
		train.out,
		epochs=max.epochs,
		shuffle=TRUE,
		validation_data= list(test.in, test.out)
		)
		
		b$val.history<-history

		
		epochs<-max(20, which(abs(history$metrics$val_loss - min(history$metrics$val_loss)) < abs(history$metrics$val_loss[1] - min(history$metrics$val_loss))*0.01)[1], na.rm=TRUE)
    
		message(c("selected epochs based on cross-validation: ", epochs))

		# re-source the model
   		source(encoder.code.file, local=TRUE)
    
		encoder.out<-my.encoder %>% 
					layer_dropout(rate = min.dropout ) %>%
					layer_dense(units =  ncomp )
   		source(decoder.code.file, local=TRUE)
        
		if(is.null(confounder))
		{
			autoencoder_model <- keras_model(inputs = input_layer, outputs = my.decoder)
    	}else{
			autoencoder_model <- keras_model(c(input_layer,auxiliary_input), outputs = my.decoder)
		}
	
    	autoencoder_model %>% compile(
    		loss='mean_squared_error',
    		optimizer='adam',
    		metrics = c('accuracy')
    	)


		history <-
		autoencoder_model %>%
		keras::fit(data.in,
		data.out,
		epochs=epochs,
		shuffle=TRUE,
		#validation_data= list(data.in, data.out)
		)
		    
		b$fit.history<-history
    
	}else{
	
		history <-
		autoencoder_model %>%
		keras::fit(list(train.in, train.confounder),
		train.out,
		epochs=max.epochs,
		shuffle=TRUE,
		validation_data= list(list(test.in, test.confounder), test.out)
		)	
		
		b$val.history<-history

		
		epochs<-max(20, which(abs(history$metrics$val_loss - min(history$metrics$val_loss)) < abs(history$metrics$val_loss[1] - min(history$metrics$val_loss))*0.01)[1], na.rm=TRUE)
 
		message(c("selected epochs based on cross-validation: ", epochs))
 
		# re-source the model
   		source(encoder.code.file, local=TRUE)
    
		encoder.out<-my.encoder %>% 
					layer_dropout(rate = min.dropout ) %>%
					layer_dense(units =  ncomp )
   		source(decoder.code.file, local=TRUE)
        
		if(is.null(confounder))
		{
			autoencoder_model <- keras_model(inputs = input_layer, outputs = my.decoder)
    	}else{
			autoencoder_model <- keras_model(c(input_layer,auxiliary_input), outputs = my.decoder)
		}
	
    	autoencoder_model %>% compile(
    		loss='mean_squared_error',
    		optimizer='adam',
    		metrics = c('accuracy')
    	)

 
		history <-
		autoencoder_model %>%
		keras::fit(list(data.in, confounder),
		data.out,
		epochs=epochs,
		shuffle=TRUE,
		#validation_data= list(test.in, test.out)
		)	
		
		b$fit.history<-history

	}
	
	
	#############
	
    autoencoder_weights <- autoencoder_model %>% keras::get_weights()
    
    filepath=paste(temp.path,'/autoencoder_weights.hdf5',sep='')
    
    keras::save_model_weights_hdf5(object = autoencoder_model,filepath = filepath,overwrite = TRUE)
    
    encoder_model <- keras_model(inputs = input_layer, outputs = encoder.out)
    
    
    try(encoder_model %>% keras::load_model_weights_hdf5(filepath = filepath, skip_mismatch = TRUE), silent=TRUE)
    try(encoder_model %>% keras::load_model_weights_hdf5(filepath = filepath, skip_mismatch = TRUE, by_name=TRUE), silent=TRUE) 
	### by_name=TRUE works in windows, but not in Mac. Need to delete when using Mac.
    
    encoder_model %>% compile(loss='mean_squared_error', optimizer='adam', metrics = c('accuracy'))
    
    embeded_points <- encoder_model %>% keras::predict_on_batch(x = data.in)
    
    b$embeded<-as.matrix(embeded_points)
    #b$weights<-autoencoder_weights
    
    #### find feature importance
    
    influ<-rep(0, ncol(data.in))
	if(pairwise.importance) influ.y<-matrix(0, ncol=ncol(data.in), nrow=ncol(data.out))
    
    if(importance.permutations>0)
    {
        this.data<-data.in
        pred0 <- predict (object = encoder_model, x = this.data)
		
		if(pairwise.importance)
		{
			if(is.null(confounder))
			{
				predy0<- predict (object = autoencoder_model, x = this.data)
			}else{
				predy0<- predict (object = autoencoder_model, x = list(this.data, confounder))
			}
		}
		
        for(i in 1:ncol(data.in))
        {
            this.data<-data.in
            rec<-rep(0, importance.permutations)
			if(pairwise.importance) rec2<-matrix(0, ncol=importance.permutations, nrow=ncol(data.out))
            for(n in 1:importance.permutations)
            {
                this.data[,i]<-sample(this.data[,i], nrow(this.data))
                pred <- predict (object = encoder_model, x = this.data)
				if(pairwise.importance)
				{
					if(is.null(confounder))
					{
						predy<- predict (object = autoencoder_model, x = this.data)
					}else{
						predy<- predict (object = autoencoder_model, x = list(this.data, confounder))
					}
				}
						
                rec[n]<-sum(sqrt(apply((pred-pred0)^2,1,sum)))
				if(pairwise.importance) rec2[,n]<-apply((predy-predy0)^2,2,sum)
            }
            influ[i]<-mean(rec)
			if(pairwise.importance) influ.y[,i]<-apply(rec2,1,mean)
            if(i %% round(ncol(data.in)/10) == 0) cat(i, " ")
        }
    }
    
    b$imp<-influ
    if(pairwise.importance) b$pair.imp<-influ.y

    k_clear_session()
    
    return(b)
}
