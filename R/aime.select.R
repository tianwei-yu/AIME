aime.select <-
function(data.in, data.out, confounder=NULL, all.in.layers=1:5, all.out.layers=1:5, all.dropouts=c(0.2, 0.3, 0.4, 0.5), repeats=3, do.normalize=TRUE, test.proportion= 0.2, ncomp=4, min.dropout=0.05, flat.dropout=FALSE, activation="relu", max.epochs=100, cor.cut=0.75, kurtosis.cut=0.25, skew.cut=0.25, col="black", cex=0.5)
{	
	combos<-expand.grid(all.in.layers, all.out.layers, all.dropouts)
	rec<-new("list")
	rec.skew<-1:nrow(combos)
	rec.kurt<-1:nrow(combos)
	rec.cor<-1:nrow(combos)

	for(m in 1:nrow(combos))
	{
		message("working on setting ", m, " of ", nrow(combos))
		is.there<-dir(pattern=paste("select",combos[m,1], combos[m,2], combos[m,3], ".bin"))
		if(length(is.there)==0)
		{
			b<-new("list")
			for(k in 1:repeats)
			{
				b[[k]]<-aime(data.in=data.in, data.out=data.out, confounder=confounder, do.normalize=do.normalize, test.proportion= test.proportion, ncomp=ncomp, max.dropout=combos[m,3], min.dropout=min.dropout, flat.dropout=flat.dropout, in.layers=combos[m,1], out.layers=combos[m,2], encoder.layer.sizes=NA, decoder.layer.sizes=NA, activation=activation, max.epochs=max.epochs, importance.permutations=0)
				b[[k]]<-b[[k]][-4:-5]
			}

			r<-new("list")
			r2<-new("list")
			for(k in 1:repeats)
			{
				r[[k]]<-mean(abs(apply(b[[k]]$embeded,2,kurtosis)))
				r2[[k]]<-mean(as.dist(abs(cor(b[[k]]$embeded))))
			}


			this<-list(b=b, r=r, r2=r2)
			save(this, file=paste("select",combos[m,1], combos[m,2], combos[m,3], ".bin"))
		}
	}

	for(m in 1:nrow(combos))
	{
		load(paste("select",combos[m,1], combos[m,2], combos[m,3], ".bin"))
		
		curr<-rep(0,0)

		for(k in 1:repeats)
		{
			this$r[[k]]<-mean(abs(apply(this$b[[k]]$embeded,2,kurtosis)))
			this$b[[k]]<-this$b[[k]][-4:-5]

			curr.skew.kurt<-c(0,0)
			try(curr.skew.kurt<-as.numeric(as.vector(mvn(this$b[[k]]$embeded)$multivariateNormality[1:2,2])))
			curr<-rbind(curr, curr.skew.kurt)
		}
		#print(curr)
		#curr<-curr[-1,]
		rec[[m]]<-this$b
		rec.skew[m]<-mean(curr[,1],na.rm=T)
		rec.kurt[m]<-mean(curr[,2],na.rm=T)
		rec.cor[m]<-mean(unlist(this$r2),na.rm=T)
		rm(this)
		gc()
	}

	combos<-cbind(combos, rank(rec.skew)/length(rec.skew), rank(rec.kurt)/length(rec.kurt), rec.cor)
	combos<-combos[which(combos[,4]>skew.cut & combos[,5]>kurtosis.cut & combos[,6]<cor.cut),]
	colnames(combos)[1:3]<-c("in layers","out layers","dropout")

	if(nrow(combos)>0)
	{
		pdf("aim selection.pdf")
		for(m in 1:nrow(combos))
		{
			load(paste("select",combos[m,1], combos[m,2], combos[m,3], ".bin"))
			for(k in 1:length(this$b)) pairs(this$b[[k]]$embeded, cex=cex, col=col, main=paste("select",combos[m,1], combos[m,2], combos[m,3],"kurtosis:",mean(abs(apply(this$b[[k]]$embeded,2,kurtosis)))))
		}
		dev.off()
	}
	return(combos)
}
