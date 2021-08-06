# AIME
AIME Autoencoder-based Integrative Multi-omics data Embedding

AIME extracts data representation for omics data integrative analysis. The method can adjust for confounder variables, achieve informative data embedding, rank features in terms of their contributions, and find pairs of features from the two data types that are related to each other through the data embedding. 

A github install by using devtools:install_github(AIME) will automatically invoke the installation of the keras package for R. However the first time you run AIME, and error may pop up, asking you to install tensorflow. Simply follow the command in the error message to install tensorflow. The AIME can be run in CPU mode. Running in GPU mode requires the installation of GPU version of tensorflow. 

The method takes two matrices as inputs: the input matrix X_(N×p), and the output matrix Y_(Nxq). It also takes a confounder matrix (could be NULL if not necessary)  C_(N×s).
With regard to the sizes of the layers of the network, the method allows three different ways for the user to specify. (1) The user can directly specify the sizes of all the individual layers; (2) the user can input a shrinkage factor, such that the size of each layer in the encoder is the product of the size of the previous layer and the shrinkage factor, and the size of each decoder layer is the product of the next layer and the shrinkage factor; (3) the user can input the desired number of input/out layers, and the shrinkage factor is calculated based on the number of layers. 

Here is a small example:

```{r example}
X<-matrix(rnorm(50000), ncol=50)
Y<-matrix(rnorm(70000), ncol=70)
X2<-X

for(i in 1:20) 
{
	relat<-sample(1:3,1)
	if(relat==1) Y[,i]<-(X[,1]+X[,2]+X[,3]+X[,4])^2
	if(relat==2) Y[,i]<-sin((X[,1]+X[,2]+X[,3]+X[,4])*pi)
	if(relat==3) Y[,i]<-abs(X[,2]+X[,3]+X[,4])
	X2[,i]<-X[,ceiling(i/5)]+rnorm(nrow(X2))*0.5
}
X<-X2
for(i in 1:ncol(Y)) Y[,i]<- (Y[,i]-mean(Y[,i]))/sd(Y[,i])

b<-aime(data.in=X, data.out=Y, layer.shrink.factor=4, max.dropout=0.25, max.epochs=100,importance.permutations=5, ncomp=3)

plot(b$imp)
pairs(b$embeded)

```


g<-aime.select(data.in=X[,-1:-5], data.out=Y, confounder=X[,1:10],all.in.layers=3:4, all.out.layers=3:4, all.dropouts=c(0.25, 0.5), repeats=2)



