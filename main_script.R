library(devtools)
library(ggplot2)
library(ggbiplot)
library(corrplo)
library(party)
library(dplyr)
library(FactoMineR)
library(factoextra)
library(randomForest)
#renv::snapshot()

nba = read.csv("NBA_Dataset.csv")

# dimenzije ucitanog dataseta - 17697 redova x 52 kolona (ukupno)
dim(nba)
str(nba)
head(nba)

# uzimamo samo bitnije kolone; ovo ce se zapravo uz pomoc PCA metode uraditi
# nba = nba[c("pos","age","g", "gs", "mp_per_g", "fg_per_g", "fg3a_per_g", "fg2a_per_g", "ft_per_g", "ast_per_g")]

# proveravamo da li postoje NA vrednosti - nema NA vrednosti
colSums(is.na(nba))
sum(is.na(nba))

# koliko redova mozemo da iskoristimo - svi redovi
sum(complete.cases(nba))

# provera da li postoji veza izmedju vremena provedenog u igri i pogodataka po igri - vidi se korelacija
ggplot1 = ggplot(nba, aes(mp_per_g, fg_per_g)) + geom_point(colour = "skyblue", alpha = 0.3) + theme(axis.title = element_text(size = 8.5))
ggplot1 + geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE)

# provera da li postoji veza izmedju vremena provedenog u igri i broja asistencija po igri
ggplot2 = ggplot(nba, aes(mp_per_g, ast_per_g)) + geom_point(colour = "skyblue", alpha = 0.3) + theme(axis.title = element_text(size = 8.5))
ggplot2

# provera da li postoji veza izmedju pogodataka po igri i broja asistencija po igri
ggplot3 = ggplot(nba, aes(fg_per_g, ast_per_g)) + geom_point(colour = "skyblue", alpha = 0.3) + theme(axis.title = element_text(size = 8.5))
ggplot3

# provera da li postoji veza izmedju pokusaja za 2 poena po igri i broja pokusaja za 3 poena po igri
ggplot4 = ggplot(nba, aes(fg2a_per_g, ft_per_g)) + geom_point(colour = "skyblue", alpha = 0.3) + theme(axis.title = element_text(size = 8.5))
ggplot4

# field goals per game (fg_per_g) - poeni po igri
ggplot5 = ggplot(nba, aes(fg_per_g)) + geom_bar(fill = "skyblue")
ggplot5

# godine igraca
ggplot6 = ggplot(nba, aes(age)) + geom_bar(fill = "skyblue")
ggplot6

# pozicija na kojoj igraju
ggplot0 = ggplot(nba, aes(pos)) + geom_bar()
ggplot0

# skaliranje (nije potrebno ako se radi PCA uz pomoc FactoMineR)
# nba$fg_per_g <- scale(nba$fg_per_g, center= TRUE, scale=TRUE)

# korelacija izmedju svih ukljucenih kolona [-1 (crveno), 1 (tamno plavo)]
nba = nba %>% select(- c(X, award_share)) # ukloni kolone koje se ne koriste (po predlogu izvora dataseta)
train_num=nba%>%select_if(is.numeric)
corMatrix=cor(train_num)
ggplot7 = corrplot(corMatrix,order = "FPC",method = "color",type = "lower", tl.cex = 0.6, tl.col = "black")
ggplot7

# Odraditi Principal Component Analysis
res.pca <- PCA(train_num, graph = FALSE, ncp = 5, scale.unit=TRUE) # ncp je broj dimenzija koje ce biti u krajnjem rezultatu
summary(res.pca)
print(res.pca)

# Extract the eigenvalues/variances of principal components
# The eigenvalues measure the amount of variation retained by each principal component
eig.val = get_eigenvalue(res.pca)
eig.val

# Visualize the eigenvalues; Produce scree plot using the fviz_eig() or fviz_screeplot() [factoextra package]
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))

# PCA Extract the results for individuals and variables
ind <- get_pca_ind(res.pca)
ind
var <- get_pca_var(res.pca)
var
head(var$cor)

# korelacioni krug; Visualize the results individuals and variables
fviz_pca_ind(res.pca, col.ind = "skyblue")
fviz_pca_var(res.pca, col.var = "black")
# lepsi prikaz varijabli
fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# biplot of individuals and variables
fviz_pca_biplot(res.pca, col.ind = "skyblue", col.var = "black")

# kvalitet reprezentacije promenljivih na mapi faktora (cos2 - kvadrat kosinusne funkcije, kvadrat vrednosti)
corrplot(var$cos2, is.corr=FALSE)

# Total cos2 of variables on Dim.1 and Dim.2 (vece vrednosti = bitnija promenljiva/dimenzija)
fviz_cos2(res.pca, choice = "var", axes = 1:2)

# udeo promenljivih u PC1 (Dim1), PC2 (Dim2), ...
fviz_contrib(res.pca, choice = "var", axes = 1, top = 10)
fviz_contrib(res.pca, choice = "var", axes = 2, top = 10)
fviz_contrib(res.pca, choice = "var", axes = 3, top = 10)
fviz_contrib(res.pca, choice = "var", axes = 4, top = 10)
fviz_contrib(res.pca, choice = "var", axes = 5, top = 10)

head(var$contrib)


#################

# pronadji promenljive koje imaju najvecu udeo u dimenzijama navedenim u axes=c(), posmatraj prvih top = X varijabli, sortiraj
f <- factoextra::fviz_contrib(res.pca, choice = "var", axes = c(1,2), top = 100, sort.val = c("desc"))
f

# save data from contribution plot
dat <- f$data
dat

# filter out ID's that are higher than 2 // vidi se sa grafikona
r <- rownames(dat[dat$contrib>2.1,])

# extract these from your original data frame into a new data frame for further analysis
df <- train_num[r]
df
dim(df)
head(df)

##################


# Machine Learning algoritam (random forest)

# postavi random seed (potrebno za data_delimiter)
set.seed(2)

# delimo dataset na trening, test i validacioni (odnos 70:30)
df$fg_per_g = round(df$fg_per_g)
data_delimiter <- sample( c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.7,0.3) )

nba_training = df[data_delimiter,]
nba_test = df[!data_delimiter,]
nba_validation = df[!data_delimiter,]

nba_test = nba_test %>% select(- c(fg_per_g))

# pregled promenljivih
head(nba_training)
head(nba_test)
head(nba_validation)

# primena random forest algoritma
rf <- randomForest(as.factor(fg_per_g) ~ ., data=nba_training, importance=TRUE, proximity=TRUE, ntree=500)
rf

nba_test$fg_per_g = predict(rf, newdata = nba_test)

# tabela konfuzione matrice
cm = table(nba_test$fg_per_g, nba_validation$fg_per_g )
cm

# tumacenje tabele, tj. procenat tacnosti prilikom predvidjanja
p = mean(nba_test$fg_per_g == nba_validation$fg_per_g)*100
cat("Procenat tacnosti predvidjanja: ", p, "%.")
