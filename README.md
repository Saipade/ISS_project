

ˇ ´

Fakulta informacnıch technologiı

Vysoke ucenı technicke v Brne

´

´ ˇ ´

´

ˇ

Sign´aly a syst´emy

Projekt 2021/22

Maksim Tikhonov (xtikho00)

\7. ledna 2022





´

1 Uvod

1.1 Struktura projektu

Ve adres´aˇri /src se nach´az´ı soubor project.ipynb, ktery´ obsahuje zdrojovy´ kod analyzuj´ıc´ı a

vyˇcist’uj´ıc´ı pomoc´ı ﬁltr˚u vstupn´ı sign´al ze souboru

<https://www.fit.vutbr.cz/study/courses/ISS/public/proj2021-22/signals/xtikho00.wav>[ ](https://www.fit.vutbr.cz/study/courses/ISS/public/proj2021-22/signals/xtikho00.wav)a

generuj´ıc´ı grafy pouˇzit´e v tomto protokolu.

Ve adres´aˇri /audio se nach´azej´ı audio soubory, kter´e jsou vy´sledky pr´ace programu.

1.2 Pouˇzit´e prostˇred´ı

Projekt byl implementov´an v jazyce Python na z´akladˇe platformy Google Colab.

ˇ

2 Reˇsen´ı projektu

2.1 Standardn´ı zad´an´ı

2.1.1 Z´aklady

Prvotn´ı analy´za vstupn´ıho sign´alu (stanoven´ı d´elky v sekund´ach, nalezen´ı max. a min. hodnot

sign´alu) byla implementov´ana pomoc´ı z´akladn´ıch funkc´ı Pythonu .min a .max pro extr´emn´ı

hodnoty sign´alu a dˇelen´ım d´elky signa´lu ve vzorc´ıc´ıch vzorkovac´ı frekvenc´ı pro nalezen´ı d´elky sign´alu

v sekund´ach.

Data

Soubor

D´elka

Hodnota

Max.

[s]

[ ]

Min.

Stˇr.

xtikho00.wav 3.28325 52532 -0.1983642578125 0.23193359375 -2.5046350096726756e-5

Tabulka 1: Z´akladn´ı u´daje o sign´alu

Obr´azek 1: Vstupn´ı sign´al

1





2.1.2 Pˇredzpracov´an´ı a r´amce

Pˇredzpracov´an´ı sign´alu (vylouˇcen´ı stejnosmˇern´eho proudu a pˇrevod do rozsahu -1 aˇz +1) bylo

provedeno prostˇrednictv´ım odeˇc´ıt´an´ı jeho stˇredn´ı hodnoty a dˇelen´ım maxim´em jeho absolutn´ı

hodnoty.

Pak pro rozdˇelen´ı p˚uvodn´ıho sign´alu do r´amc˚u, na jeho konec byly pˇrida´ny vzorky s

hodnotou nula tak, aby jeho d´elka byla dˇeliteln´a d´elkou r´amc˚u - 1024.

Pro generov´an´ı matic r´amc˚u byla pouˇzita funkce np.tile. Vyˇslo mi celkem 102 r´amc˚u.

def frame\_signal ( signal = None, frameLength = 1024, overlapLength = 512 ) :

signalLength = len(signal)

stepLength = frameLength - overlapLength

numberOfFrames = np.abs(signalLength - overlapLength) // np.abs(frameLength - overlapLength) + 1

restLength = np.abs(signalLength - overlapLength) % np.abs(frameLength - overlapLength)

addedZeroSignalLength = stepLength - restLength

addedZeroSignal = np.zeros(addedZeroSignalLength)

signal = np.append(signal, addedZeroSignal)

index1 = np.tile(np.arange(0, frameLength), (numberOfFrames, 1))

index2 = np.tile(np.arange(0, numberOfFrames \* stepLength, stepLength), (frameLength, 1)).T

indices = index1 + index2

return signal[indices.astype(np.int32, copy=False)]

Po prohlednut´ı graf˚u vˇsech tˇechto r´ameˇck˚u jsem si vybral ten nejv´ıce znˇej´ıc´ı r´amec ˇc´ıslo 25

reprezentuj´ıc´ı souˇc´ast fon´emu <a> ve slovˇe dark. Krit´erii nejvhodnˇejˇs´ı znˇelosti jsou periodick´a

pˇr´ıroda samotn´eho sign´alu a to, ˇze zvuk sign´alu reprezentuje nˇejakou samohl´asku.

Obr´azek 2: Ra´mec ˇc´ıslo 25

2





2.1.3 DFT

Diskr´etn´ı Fourierova Transformace je implementov´ana dle vzorku:

NX−1

2πnk

N

X[k] =

x[n] · e−j

n=0

























1

1

. . .

1

X[0]

X[1]

x[0]

x[1]

−j 2π(N

−1023)(k−1023)

. . . e−j

2π(N−1023)(k−1)

1 e







N

N











.

 = .

.

.

· 

.





.



.

.

.

.

.

.





.





.

.

.

.



.

X[k − 1]

e−j 2π(N

−1023)(k−1)

e−j

2π(N−1)(k−1)

x[N 1]

−

1

. . .

N

N

Funkce myDFT generuje matice tvaru k × N, kde N = k = 1024, a vektorovˇe n´asob´ı ji p˚uvodn´ım

sign´alem (r´amcem). Vy´sledkem je jednorozmˇerny´ vektor koeﬃcient˚u DFT.

def myDFT ( signal = None, numberOfSamples = 1024 ) :

\# generate matrix

serieDeFourier = np.fromfunction( lambda n, k : np.exp(-1j \* 2 \* np.pi \* n \* k / (numberOfSamples)),

(numberOfSamples, numberOfSamples),

dtype = "complex128" )

\# dot product

return np.dot(serieDeFourier, signal)

Ohlednˇe rychlosti funkce, mysl´ım, ˇze lze ˇr´ıci, ˇze je dostateˇcnˇe rychl´a (131 ms) - nejsou tam ˇz´adn´e

cykly, ale jenom n´asoben´ı vektor˚u. Oˇcevidnˇe, ˇze nejde o ˇz´adn´e rychlostn´ı srovn´an´ı s algoritmem FFT

z knihovny NumPy (131 ms vs. 16.3 µs). Zase dle funkce np.allclose vy´sledky jsou pˇribliˇznˇe stejn´e.

N´ıˇz e je graf, ktery´ pˇredstavuje vy´sledky tˇechto dvou algoritm˚u (rozd´ıl nen´ı viditelny´)

Obr´azek 3: DFT

3





2.1.4 Spektrogram

Pro vy´poˇcet spektrogramu signa´lu a pro spr´avn´e zobrazen´ı ˇcasu a frekvence na os´ach

spektrogramu byla implementov´ana funkce spectrogram, kter´a vrac´ı pole ˇcasu t frekvence f a

prvn´ı polovinu koeficient˚u DFT (jsou symetrick´e vzhledem ke stˇredu, nepotˇrebujeme zobrazovat

sign´al dvakr´at)

def spectrogram ( signal = None, frameLength = 1024, overlapLength = 512, sRate = 16000 ) :

frames = frame\_signal( signal, plotGraph = 'No' )

dftCoefficients = np.array(list(map( np.fft.fft, frames )))

dftCoefficients = dftCoefficients.T

dftCoefficients = dftCoefficients[:len(dftCoefficients) // 2]

spectreCoefficients = 20 \* np.log10(np.abs(dftCoefficients))

t = np.linspace(0, len(signal)/sRate, spectreCoefficients.shape[1])

f = np.linspace(0, 8000, spectreCoefficients.shape[0])

return t, f, spectreCoefficients

Funkce utilizuje pˇredem deﬁnovanou funkce frame signal pro rozdˇelen´ı vstupn´ıho sign´alu do

r´amc˚u, kter´e pak budou konvertova´ny na koeficienty DFT. Potom z vy´ˇse uvedeny´ch d˚uvod˚u usek´a

z´ıskan´e pole DFT koeficient˚u a upravuje jeho podle vzorku

2

P[k] = 10 log |X[k]| = 20 log |X[k]|

10

10

Obr´azek 4: Spektrogram p˚uvodn´ıho sign´alu

4





2.1.5 Urˇcen´ı ruˇsivy´ch frekvenc´ı

Ze spektrogramu p˚uvodn´ıho sign´alu je jasnˇe vidˇet frekvence ˇctyˇrech cosinusovek, m˚uj pˇredpoklad

byl n´asleduj´ıc´ı:

f = f = 750 Hz ≈ 48. koeﬁcient

1

1

f = 2f = 1500 Hz ≈ 96. koeﬁcient

2

1

f = 3f = 2250 Hz ≈ 144. koeﬁcient

3

1

f = 4f = 3000 Hz ≈ 196. koeﬁcient

4

1

Pro ovˇeˇren´ı t´eto teorie jsem implementoval funkce find bad frequencies. Poprve jsem zkusil to

ovˇeˇrit pomoc´ı vyhledav´an´ı index˚u s nejmenˇs´ımi amplitud hodnot uvnitˇr jednotlivy´ch koeﬁcient˚u

spektra (dle pˇredpokladu museli by´t 48, 96, 144, 192). Z´ıskan´e indexy m˚uj pˇredpoklad potvrdily jen

ˇc´asteˇcnˇe:

[192 144 96 191 193 145 48 196]

def find\_bad\_frequencies ( signal = None, sRate = 16000 ) :

\_, \_, spectre = spectrogram( signal )

maxs = spectre.max(axis = 1)

mins = spectre.min(axis = 1)

differences = maxs - mins

bad\_indeces = differences.argsort()[:8]

...

Druhou a mnohem pˇresnˇejˇs´ı metodou bylo z´ısk´an´ı frekvenc´ı cosinusovek z ra´mcu, ktery´

neobsahuje ˇzadn´e jin´e frekvence (ˇreˇc). Takovy´m r´amcem je prvn´ı (nulovy´). Pomoc´ı DFT z´ısk´ame to

na jaky´ch frekvenc´ıch je v tomto r´amci sign´al. Po useknut´ı poloviny koeﬁcientu (nechceme z´ıskat

stejn´e frekvence dvakr´at) najdeme pomoc´ı np.argpartition indexy (a tedy i frekvence)

cosinusovek. Vy´sledek odpov´ıda´ m´emu pˇredpokladu: [48 96 144 192]

...

frames = frame\_signal( signal, plotGraph = 'No' )

dftCoefficients = np.abs(np.fft.fft( frames[0] ))

plot\_single\_signal( dftCoefficients, xSize = 20, ySize = 5, plotLabel = 'First frame',

xLabel = 'Frequency [Hz]', DFT = 'yes', color = 'g' )

dftCoefficients = dftCoefficients[:len(dftCoefficients)//2]

badIndeces = np.argpartition(dftCoefficients, -4)[-4:]

\# Sort by value

badIndeces[0], badIndeces[1], badIndeces[2], badIndeces[3] = \

badIndeces[1], badIndeces[2], badIndeces[3], badIndeces[0]

return badIndeces \* 16000 // 1024

5





Obr´azek 5: Prvn´ı r´amec

2.1.6 Generov´an´ı sign´alu

Generovany´ sign´al se sklad´a z ˇctyˇrech cosin˚u s frekvencemi f , f , f , f = 750 Hz, 1500 Hz, 2250

1

2

3

4

Hz, 3000 Hz. Jednotliv´e cosinusovky byly vygenerov´any funkc´ı np.cos a vy´sledny´ signal je jejich

souˇctem. Frekvence vygenerovan´eho sign´alu koresponduj´ı frekvenc´ım cosin˚u z p˚uvodn´ıho sign´alu.

Vygenerovany´ sign´al je uloˇzen v souboru audio/4cos.wav

Obr´azek 6: Vygenerovany´ sign´al

6





ˇ

2.1.7 Cistic´ı ﬁltr

ˇ

Cistic´ı ﬁltr byl implementovan jako 4 p´asmov´e z´adrˇz´ı.

Filtry

0

1

2

3

4

5

6

7

8

b

0.9378 -7.1796 24.3628 -47.8379 59.4337 -47.8379 24.3628 -7.1795 0.9378

1.0000 -7.5328 25.1521 -48.5983 59.4157 -47.0628 23.5878 -6.8411 0.87949

0.9366 -6.2302 19.2874 -35.9200 43.8646 -35.9200 19.2874 -6.2302 0.9366

1.0000 -6.5430 19.9245 -36.5014 43.8495 -35.3254 18.6614 -5.9308 0.8772

0.9361 -4.7510 12.7868 -21.9019 26.1276 -21.9019 12.7868 -4.7510 0.9361

1.0000 -4.9915 13.2121 -22.2580 26.1170 -21.5354 12.3681 -4.5209 0.8763

0.9358 -2.8650 7.0326 -10.2735 12.5147 -10.2735 7.0326 -2.8650 0.9358

1.0000 -3.0108 7.2670 -10.4407 12.5081 -10.1001 6.8006 -2.7256 0.8758

Tabulka 2: Koeﬁcienty ﬁltr˚u

Hf1

Hf2

Hf3

Hf4

a

b

a

b

a

b

a

Obr´azek 7: Impulsn´ı odezvy ﬁltr˚u

Kaˇzdy´ z ﬁltr˚u m´a ˇs´ıˇr i z´avˇern´eho p´asma (W ) 30 Hz, ˇs´ıˇr i propustn´eho p´asma je (W ) 50 Hz na kaˇzd´e

s

ze stran z´avˇern´eho p´asma. Zvllnˇen´ı (ripple) je 3 dB, potlaˇcen´ı (attenuation) je 40 dB.

p

for i in range(len(frequencies)):

Wp = np.array([frequencies[i] - 65, frequencies[i] + 65]) \* 2 / sRate

Ws = np.array([frequencies[i] - 15, frequencies[i] + 15]) \* 2 / sRate

n, wNorm = sig.buttord( Wp, Ws, 3, 40 )

b[i], a[i] = sig.butter( n, wNorm, btype = 'bandstop' )

2.1.8 Nulov´e body a p´oly

Obr´azek 8: Nulov´e body a p´oly ﬁltr˚u

Z graf˚u je vidˇet, ˇze vˇsechny 4 ﬁltry jsou stabiln´ı (vˇsechny nuly a p´oly l eˇz´ı pˇr´ımo na jednotkov´e

kruˇznici). To potvrzuje i kontrola pˇrevzata z pˇr´ıklad˚u Katky Zmol´ıkov´e.

7





2.1.9 Frekvenˇcn´ı charakteristika

Obr´azek 9: Nulov´e body a p´oly ﬁltr˚u

Z graf˚u modul˚u frekvenˇcn´ı charakteristiky ﬁltr˚u je zˇrejm´e, ˇze kaˇzdy´ z nich odpov´ıd´a kaˇzd´e z

ruˇsivy´ch frekvenc´ı z p˚uvodn´ıho sign´alu (750, 1500, 2250 a 3000 Hz)

8





2.1.10 Filtrace

Filtrace sign´alu probˇehla ˇctyˇrikr´at (kaˇzdy´m z ﬁltr˚u) hned po vytvoˇren´ı jejich koeﬁcient˚u a

vzhledem k zvuku vy´sledn´eho sign´alu si mysl´ım, ˇze ﬁltrov´an´ı bylo u´spˇeˇsn´e.

Bˇehem prvn´ıch nˇekolika milisekund je ale signa´l st´ale zaˇsumˇeny´. D˚uvodem je takzvany´ ”Edge

eﬀect”, kv˚uli kter´emu se ﬁltr aplikuje na prvn´ıch nˇekolik vzork˚u.

Obr´azek 10: Filtrovany´ sign´al

Obr´azek 11: Filtrovany´ sign´al

9





Reference

[1] Rozdˇelen´ı sign´alu na r´amce: [https://superkogito.github.io/blog/SignalFraming.html#](https://superkogito.github.io/blog/SignalFraming.html#signal-framing)

[signal-framing](https://superkogito.github.io/blog/SignalFraming.html#signal-framing)

[2] Vykreslov´an´ı signa´lu, frekvenˇcn´ı charakteristika sign´alu: [https://nbviewer.org/github/](https://nbviewer.org/github/zmolikova/ISS_project_study_phase/tree/master/)

[zmolikova/ISS_project_study_phase/tree/master/](https://nbviewer.org/github/zmolikova/ISS_project_study_phase/tree/master/)

[3] SciPy dokumentace: <https://docs.scipy.org/doc/>

[4] NumPy dokumentace: <https://numpy.org/doc/>

10


