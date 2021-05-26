\name{aime.select}
\alias{aime.select}
\title{
Parameter selection for AIME
}
\description{
Scan through some parameter settings for the selection of optimal combination.
}
\usage{
aime.select(data.in, data.out, confounder=NULL, all.in.layers=1:5, all.out.layers=1:5, all.dropouts=c(0.2, 0.3, 0.4, 0.5), repeats=3, do.normalize=TRUE, test.proportion= 0.2, ncomp=4, min.dropout=0.05, flat.dropout=FALSE, activation="relu", max.epochs=100, cor.cut=0.75, kurtosis.cut=0.25, skew.cut=0.25, col="black", cex=0.5)
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
confounder data matrix, with variables in the columns
}
  \item{all.in.layers}{
All in.layers values to be tested
}
  \item{all.out.layers}{
All out.layers values to be tested
}
  \item{all.dropouts}{
All dropout rates to be tested
}
  \item{repeats}{
Number of repeats at each parameter setting
}
  \item{do.normalize}{
whether to normalize each variable to the same variance
}
  \item{test.proportion}{
proportion of samples to use as testing data
}
  \item{ncomp}{
number of hidden components to seek, i.e. the dimension of the embeded data
}
  \item{min.dropout}{
The minimum dropout rate if flat.dropout is FALSE. In this case the dropout rate at each parameter setting is used as maximum dropout.
}
 \item{flat.dropout}{
If TRUE, a single dropout rate is used on all layers. The value of max.dropout is used across all layers. If FALSE, outer layers have higher dropout rate, and inner layers have smaller dropout rate.
}
  \item{activation}{
The activation function to be passed to tensorflow
}
  \item{max.epochs}{
A validation process is used to find the optimal number of epochs. Max.epochs specifies the maximumn number of epochs to try
}
  \item{cor.cut}{
The threshold for average absolute correlations between the embeded dimensions. If a setting yields average correlation larger than this value, it is ignored.
}
  \item{kurtosis.cut}{
The threshold for kurtosis of the embeded data. If a setting yields average correlation smaller than this value, it is ignored.
}
  \item{skew.cut}{
The threshold for skewness of the embeded data. If a setting yields average correlation smaller than this value, it is ignored.
}
  \item{col}{
The vector of data point colors when generating the plots in PDF format.
}
  \item{cex}{
The vector of data point sizes when generating the plots in PDF format.
}
}
\details{
The subroutine scans through the combinations of in.layers, out.layers, and dropout rates, and reports the kurtosis, skewness, and average correlations between the embeded dimensions, for the user to make a decision which parameter combination to use.
}
\value{
A PDF file is generated in the working folder with plots of embedded data. A matrix is returned, with columns: in.layers, out.layers, dropout, normalized rank of kurtosis, normalized rank of skewness, average correlation between embedded dimensions.
}

\author{
Tianwei Yu <yutianwei@cuhk.edu.cn>
}
\seealso{
aime
}
\examples{
\dontrun{
X<-rmvnorm(1000, mean=rep(0,20), sigma=diag(20))
Y<-rmvnorm(1000, mean=rep(0,30), sigma=diag(30))
for(i in 1:5) Y[,i]<-abs(X[,i])+0.5*rnorm(100)
for(i in 6:10) Y[,i]<-sin(X[,i-5])+0.5*rnorm(100)
for(i in 11:15) Y[,i]<-X[,i-10]^2+0.5*rnorm(100)
g<-aime.select(X,Y,all.in.layers = 3:5, all.out.layers = 3:4, all.dropouts = c(0.3,0.4), repeats = 2, cor.cut=0.85, kurtosis.cut=0.1, skew.cut=0.1)
g
}
}
\keyword{Model selection}

