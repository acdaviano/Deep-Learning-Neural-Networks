#we load the libraries that I will use for the exercise 
#and then change the working directory for better access.
library(jsonlite)
library(caret)
library(h2o)
library(ggplot2)
library(data.table)
library(e1071)
data = file("D:\\Regis MSDS\\Regis\\MSDS 664\\data\\train-images.idx3-ubyte.gz", "rb")
readBin(data, integer(), n=4, endian="big")
#I then read in the data the way the site mentions since the sizes are meant to be 28X28
m = matrix(readBin(data,integer(), size=4, n=28*28, endian="big"),28,28)
par(mfrow=c(2,2))
for(i in 1:4){m = matrix(readBin(data,integer(), size=1, n=28*28, endian="big"),28,28);
image(m[,28:1])}
#then we start the h2o process and change the data format to fit
h2o.init()
#we then run the program
load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}
load_label_file <- function(filename) { 
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}
imagetraining<-as.data.frame(load_image_file("train-images.idx3-ubyte.gz"))
imagetest<-as.data.frame(load_image_file("t10k-images.idx3-ubyte.gz"))
labeltraining<-as.factor(load_label_file("train-labels.idx1-ubyte.gz"))
labeltest<-as.factor(load_label_file("t10k-labels.idx1-ubyte.gz"))
imagetraining[,1]<-labeltraining
imagetest[,1]<-labeltest
Training<-imagetraining
Test<-imagetest 
sample_n<-3000
order<-sample(60000,sample_n)
training<-Training[order,]
validating<-Training[-order,]
test_x<-Test[,-1]

cl<-h2o.init(max_mem_size = "20G",nthreads = 10)
h2odigits<-as.h2o(training, destination_frame = "h2odigits")
h2odigits_v<-as.h2o(validating,destination_frame = "h2odigits_v")
h2odigits_t<-as.h2o(Test, destination_frame = "h2odigits_t")
h2odigits_train_x<-h2odigits[,-1]
h2odigits_test_x<-h2odigits_t[,-1]
xnames<-colnames(h2odigits_train_x)

system.time(hd_fnn_01<-h2o.deeplearning(
  x=xnames,y="n",training_frame = h2odigits,validation_frame = h2odigits_v,
  activation = "RectifierWithDropout",hidden=c(400,100),epochs=400,
  loss="CrossEntropy",adaptive_rate = F,
  rate=0.0001,input_dropout_ratio = c(0.2),hidden_dropout_ratios = c(0.5,0.25)
))

hd_fnn_01_results<-h2o.predict(hd_fnn_01,h2odigits_t)
results<-as.data.frame(hd_fnn_01_results[,1])
caret::confusionMatrix(unlist(results),Test$n)$overall
 Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue 
 0.9261000      0.9178560      0.9207974      0.9311528      0.1135000      0.0000000

#we then build more models with corrected parameters to see if they come out better
model2<-h2o.deeplearning(x=x,y="n",training_frame = TrainingH,validation_frame = TestH,distribution = "multinomial",activation="RectifierWithDropout",hidden = c(50,50,50),input_dropout_ratio = .2,sparse=T,epochs=100)
model3<-h2o.deeplearning(x=x,y="n",training_frame = TrainingH,validation_frame = TestH,distribution = "multinomial",activation="RectifierWithDropout",hidden = c(30,30,30),input_dropout_ratio = .2,sparse=T,epochs=50,nfolds=5)
model4<-h2o.deeplearning(x=x,y="n",training_frame = TrainingH,validation_frame = TestH,distribution = "multinomial",activation="RectifierWithDropout",hidden = c(30,30,30),input_dropout_ratio = .1,sparse=T,epochs=50,nfolds=5)
# we then look into the next predictions for the other models we want moving forward.
#model 2
head(TestH[,1])
head(results2)
caret::confusionMatrix(unlist(results2),Test$n)$overall
#model3
head(TestH[,1])
head(results3)
caret::confusionMatrix(unlist(results3),Test$n)$overall
#model4
head(TestH[,1])
head(results4)
caret::confusionMatrix(unlist(results4),Test$n)$overall