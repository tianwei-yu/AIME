# AIME
AIME Autoencoder-based Integrative Multi-omics data Embedding

AIME extracts data representation for omics data integrative analysis. The method can adjust for confounder variables, achieve informative data embedding, rank features in terms of their contributions, and find pairs of features from the two data types that are related to each other through the data embedding. 

A github install by using devtools::install_github("tianwei-yu/AIME") will automatically invoke the installation of the keras package for R. However the first time you run AIME, and error may pop up, asking you to install tensorflow. Simply follow the command in the error message to install tensorflow. AIME can be run in CPU mode. Running in GPU mode requires the installation of GPU version of tensorflow. 

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

We include a utility to select some key parameters. At each hyperparameter setting, the data embedding (matrix E) is computed, and the average absolute pairwise correlation between the columns of the E matrix is calculated. Among the settings for which the average correlation is below a threshold, the Mardia’s multivariate skewness and kurtosis coefficients are calculated for the embedded data. We rank each setting by the skewness and kurtosis of the embedded data, and then select the setting the yield the highest average rank of skewness and kurtosis. This process selects parameter settings that yield embedding that is not highly correlated, as well as with a distribution far from multivariate normal. 

Here is a small example:

```{r example}

g<-aime.select(data.in=X[,-1:-5], data.out=Y, confounder=X[,1:10],all.in.layers=3:4, all.out.layers=3:4, all.dropouts=c(0.25, 0.5), repeats=2)

```

Here is the example output from the code. Smaller ranks in skewness and kurtosis indicate non-Gaussian distributions which AIME seeks. The last column is the average absolute correlation between the embeded dimensions. Smaller is better. 

```{r example}
> g
  in layers out layers dropout rank(rec.skew)/length(rec.skew) rank(rec.kurt)/length(rec.kurt)   rec.cor
6         4          3     0.5                            0.75                           0.750 0.6794038
7         3          4     0.5                            0.50                           0.875 0.6598576
8         4          4     0.5                            1.00                           1.000 0.7493096

```

In addition, a PDF file is written to the working directory, which contains the pairwise scatterplots. The user can visually examine the plots to select the hyperparameter settings. R binary files of the results in each parameter setting are also written in the folder. 
