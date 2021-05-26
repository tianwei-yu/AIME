\name{aime}
\alias{aime}
\title{
Autoencoder-based Integrative Multi-omics data Embedding
}
\description{
The function for Autoencoder-based Integrative Multi-omics data Embedding
}
\usage{
aime(data.in, data.out, confounder = NULL, do.normalize = TRUE, test.proportion = 0.2, ncomp = 4, max.dropout = 0.4, min.dropout = 0.05, flat.dropout = FALSE, in.layers = NA, out.layers = NA, layer.shrink.factor = 2, encoder.layer.sizes = NA, decoder.layer.sizes = NA, activation = "relu", max.epochs = 100, importance.permutations = 1, pairwise.importance = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data.in}{
input data matrix, with variables in the columns
}
  \item{data.out}{
output data matrix, with variables in the columns. data.in and data.out should have the same number of rows (samples)
}
  \item{confounder}{
confounder data matrix, with variables in the columns}
  \item{do.normalize}{
whether to normalize each variable to the same variance
}
  \item{test.proportion}{
proportion of samples to use as testing data
}
  \item{ncomp}{
number of hidden components to seek, i.e. the dimension of the embeded data
}
  \item{max.dropout}{
The maximum dropout rate if flat.dropout is FALSE
}
  \item{min.dropout}{
The minimum dropout rate if flat.dropout is FALSE
}
  \item{flat.dropout}{
If TRUE, a single dropout rate is used on all layers. The value of max.dropout is used across all layers. If FALSE, outer layers have higher dropout rate, and inner layers have smaller dropout rate.
}
  \item{in.layers}{
When encoder.layer.sizes takes value of NA, the user can use in.layers to specify how many encoder layers are used. If in.layers is also set as NA, then the number of layers is determed using layer.shrink.factor
}
  \item{out.layers}{
When decoder.layer.sizes takes value of NA, the user can use out.layers to specify how many decoder layers are used. If out.layers is also set as NA, then the number of layers is determed using layer.shrink.factor
}
  \item{layer.shrink.factor}{
When neither the layer sizes nor the number of layers are specified, the program uses the layer.shrink.factor to calculate how many layers are needed and what size each layer takes. The value of layer.shrink.factor decides the relative size of an inner layer in relation to its immediate outler layer
}
  \item{encoder.layer.sizes}{
The user can use a vector to specify the number of nodes in each encoder layer, from ourer to inner layers
}
  \item{decoder.layer.sizes}{
The user can use a vector to specify the number of nodes in each decoder layer, from inner to outer layers
}
  \item{activation}{
The activation function to be passed to tensorflow
}
  \item{max.epochs}{
A validation process is used to find the optimal number of epochs. Max.epochs specifies the maximumn number of epochs to try
}
  \item{importance.permutations}{
How many of permunations to be conducted to find the importance of the variables
}
  \item{pairwise.importance}{
Whether to compute the pairwise importance between input and output variables
}
}
\details{
The method can adjust for confounder variables, achieve informative data embedding, rank features in terms of their contributions, and find pairs of features from the two data types that are related to each other through the data embedding. 
}
\value{
The returned item is a list object. It contains,
\item{val.history}{The validation fit (epoch selection) history returned by keras}
\item{fit.history}{The data fit history returned by keras}
\item{embeded}{The embeded data matrix, with a dimension of N rows by ncomp columns}
\item{imp}{The importance score in the permutations. It is a vector with a length corresponding to the number of input variables.}
\item{pair.imp}{The pairwise importance score in the permutations. It is a matrix. The rows correspond to outcome variables, and the columns correspond to the input variables.}
}
\author{
Tianwei Yu <yutianwei@cuhk.edu.cn>
}
\seealso{
aim.select
}
\examples{
X<-rmvnorm(1000, mean=rep(0,20), sigma=diag(20))
Y<-rmvnorm(1000, mean=rep(0,30), sigma=diag(30))
for(i in 1:5) Y[,i]<-abs(X[,i])+0.5*rnorm(100)
for(i in 6:10) Y[,i]<-sin(X[,i-5])+0.5*rnorm(100)
for(i in 11:15) Y[,i]<-X[,i-10]^2+0.5*rnorm(100)
b<-aime(X,Y,max.epochs=100,pairwise.importance=TRUE)
plot(b$imp)
image(b$pair.imp)
}
\keyword{neural networks}
\keyword{embedding}
\keyword{joint analysis}

