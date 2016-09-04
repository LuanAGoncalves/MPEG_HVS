# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:52:39 2015

@author: Gonçalves
@author: Navegantes
"""

import bitstring as bs
import numpy as np
#import cv2


class HuffCoDec:
    '''
    Classdoc
    '''
    
#    DCLumaTB = None
#    DCChromTB = None
#    ACLumaTB = None
#    ACChromTB = None
    
    def __init__(self, hufftables):
        
        self.DCLumaTB, self.DCChromTB, self.ACLumaTB, self.ACChromTB = hufftables
        self.DiffCat = [ (0), (1),(2,3), (4,7), (8,15), (16,31), (32,63), (64,127), (128,255), (256,511),
                         (512,1023), (1024,2047),(2048,4095), (4096,8191), (8192,16383), (16384,32767) ]
        
    def tobin(self, coef, cat):
        '''
        Gera a sequência de bits equivalênte de 'coef'.
        
        input: coef->int, cat->int
        output: sout -> string
        '''
        sout=[]                                              #Cadeia de bits de saída
        if coef == 0:
            return ''
        n = cat+1                                            #Numero de bits para gerar os binários de numb
        if coef<0:
    #       d = bs.BitStream(int=int(2**n/2)-1, length=n)    #Para bitwise de inversao dos bits
            d = bs.BitStream(int=-1, length=n)
            b = bs.BitStream(int=abs(coef), length=n)        #Gera bitstream de numb
            sout = b^d                                       #Inversão dos bits (XOR)'
        else:
            sout = bs.BitStream(int=coef, length=n)
            
        return sout.bin[1:]
    
    def fwdhuff(self, DCant, coefseq, ch):
        '''
        Gera o codigo Huffman dos coeficientes.
        
        input: DCant -> int
               coefseq -> lista.
               ch -> int
        output: (nbits, huffcode) -> tupla.
        '''
        sz, DC = 0, 0; cdwrd, ACs = 1, 1
        nbits=0; huffcode = ''
        #Gera codigo huffman do coeficiente DC
        DCcur = coefseq[DC]                   #Coeficiente DC corrente
        dif = DCcur - DCant                   #Diferença coeficiente DC atual e anterior
        cat = self.coefcat(dif)               #Categoria da diferença do coeficiente DC
        bitstr = self.tobin(dif, cat)         #Converte a diferença para representaçao binária
        if ch == 0:
            DCcode = self.DCLumaTB[cat]           #Consulta a tabela - out:(size, codeword)
        else:
            DCcode = self.DCChromTB[cat]

        DChfcode = DCcode[cdwrd] + bitstr     #Código Huffman para o coef DC - huffcode + repr binária
        nbits = DCcode[sz] + cat
        huffcode = DChfcode
        
        if len(coefseq)>1:                    #Gera codigo huffman dos coeficientes AC
            run=0
            for AC in coefseq[ACs:]:
                if AC==0:
                    run+=1                                #Conta o numero de zeros que precedem o coef
                else:
                    cat = self.coefcat(AC)                #Categoria do coef
                    bitstr = self.tobin(AC, cat)          #Representação binária
                    if ch==0:                             #Seleciona tabela luma-chroma
                        ACcode = self.ACLumaTB[(run,cat)] #Consulta tabela - out:(size, codeword)
                    else:
                        ACcode = self.ACChromTB[(run,cat)]
                    AChfcode = ACcode[cdwrd] + bitstr   #Código Huffman do coef AC
                    nbits += cat + ACcode[sz]         #Calcula numero de bits
                    huffcode += AChfcode
                    run=0
                if run==16:                           #15 zeros seguidos de um zero
                    if ch == 0:
                        huffcode += '11111111001'
                        nbits += 11
                        run=0
                    else:
                        huffcode += '1111111010'
                        nbits += 10
                        run=0
        # Se houver apenas um componente na seq(DC) adiciona o fim de bloco
        if ch == 0:
            huffcode += '1010'                            #Inclui EOB no final
            nbits += 4
        else:
            huffcode += '00'
            nbits += 2
            
        return (nbits, huffcode)
        
    def invhuff(self, seqhuff, ch):
        '''
        '''
        DCant = 0
        bscdmax = 16
        ini = 0; leng = 2;
        SZseq = len(seqhuff)
        nblocks = 0; cat = -1
        basecode = ''; coefsq = []; seqaux = []
        
        basecode = seqhuff[ini:leng]
        value=(leng, basecode)
#        print ini, leng, 'init', value

        if ch <= 0:
            tabDC = self.DCLumaTB
            tabAC = self.ACLumaTB
            cdwrdMax = 9
        else:
            tabDC = self.DCChromTB
            tabAC = self.ACChromTB
            cdwrdMax = 11
        
        while leng < SZseq:
#            print '***************************'
#            print '** (RE)INICIATE DECODING ** - leng-sizeseq', leng, SZseq
#            print '***************************'
#            print ini, leng, "Verify DC coef zero: ", value
        #TRATAMENTO COEFICIENTE DC
            cat = -1
            if tabDC[0]==value:
                #cat = 0
                magdif = 0      #MAGNITUDE DA DIFERENÇA DCi-DCi-1
                coefDC = magdif + DCant
                seqaux.append(coefDC)
                DCant = coefDC
                
                ini = leng
                if ch == 0: leng+=4
                else:       leng+=2
#                print ini, leng, 'DC zero OK', value, seqaux
            else: #ini=0  #confirma
                #VERIFICA O CODIGO BASE
                for sz in range(ini+2, ini+cdwrdMax+1):
                    leng=sz
                    basecode = seqhuff[ini:leng]
                    value=(leng-ini, basecode)
#                    print ini, leng, 'verifying basecode DC-', value
                    #Procura categoria dada a chave de valores (size, basecode)
                    for key in tabDC.iterkeys():
                        if tabDC[key]==value:
                            cat = key
#                            print 'category founded - ', cat, ':', value
                    if cat != -1: break
#                    print 'deny'
                
                ini = leng
                leng = ini + cat
                magBin = seqhuff[ini:leng]
                magInt = bs.BitStream(bin=magBin).int
#                print ini, leng, 'Verify Magnitude DC-r: ', magInt, magBin        #magInt = bs.BitStream(bin=seqhuff[ini:leng]).int
                
                if magInt == 0 and cat == 1:
                    magdif = -1
                elif magInt == -1 and cat == 1:
                    magdif = 1
                elif magInt < 0:
                    magdif = ( self.DiffCat[cat][1]+magInt )+1
                else: # magInt < -1:
                    magdif = (-1*self.DiffCat[cat][1])+magInt
                    
                ini = leng
                if ch == 0: leng = ini + 4
                else:       leng += 2
#                print ini, leng, 'DC complete-magdif', magdif, magdif + DCant
            #FIM DO ELSE          #coefDC = float(magdif + DCant)
                coefDC = magdif + DCant
                seqaux.append(coefDC)
#                print 'SEQUENCIAaux-DC: ', seqaux, magdif + DCant, basecode + magBin
                DCant = coefDC
            
            basecode = seqhuff[ini:leng]
            if (basecode == '1010') or (basecode == '00'):          #TESTA EOB 
                nblocks += 1                #SE SIM, CONTA MAIS BLOCO
                ini = leng;
                leng= ini + 2               #AJUSTA PARA PROXIMO BASECODE
                basecode = seqhuff[ini:leng]
#                print ini, leng, 'EOB founded-nblocks ', nblocks
                value = (leng-ini, basecode)
                coefsq.append(seqaux)
                seqaux = []             #Reinicia seqaux para nova sequencia
                #VOLTA PARA O INICIO - linha 122  while leng < SZseq:
            else:
            #TRATAMENTO COEFICIENTES ACs
                leng -= 2
                basecode = seqhuff[ini:leng]
                value = (leng-ini, basecode)
                run, cat = -1, -1
#                print '******************'
#                print '** AC INICIATED ** ', ini, leng, basecode
#                print '******************'
                while True: #basecode != '1010': #leng < len(seqhuff):
                #VERIFICA BASECODE
                    run, cat = -1, -1
                    for sz in range(ini+2, ini+bscdmax+1):    #ATE O MAXIMO TAMANHO DE BASECODE
                        leng = sz
                        basecode = seqhuff[ini:leng]
                        value=(leng-ini, basecode)#value=(leng-ini, basecode)
#                        print ini, leng, 'Verifying basecode AC', value
                        #Procura categoria dada a chave de valores (size, basecode)
                        for key in tabAC.iterkeys():
                            if tabAC[key]==value:
                                run, cat = key  #runcat = (run, category)
#                        print 'Basecode verified', (run, cat),':', value
                        if (run, cat) != (-1, -1):
                            break
#                        print 'Deny'
                    #SE BASECODE FOI (0,0) ENCONTROU 'EOB' -> '00' OU '1010'
                    if (run, cat) == (0, 0):
                        nblocks += 1
                        ini += value[0]
                        coefsq.append(seqaux)
                        DCant = coefDC
#                        print 'Fim de bloco - Adiciona seqaux: ', seqaux
                        seqaux = []
                    elif (run, cat) == (15, 0):
                    #ADICIONA RUN ZEROS NA SEQUENCIA
#                        print 'SEQUENCIA DE 16 ZEROS ENCONTRADA!!'
                        for zr in range(run+1):
                            seqaux.append(0)
                        ini += value[0]
                    else:
                        if run > 0:
#                            print 'Adicionando', run, 'zeros'
                            for zr in range(run):
                                seqaux.append(0)
                        
                        ini = leng
                        leng = ini + cat
#                        print ini, leng, 'Depois de verify basecode AC - magBin: ', seqhuff[ini:leng]
                        magBin = seqhuff[ini:leng]
                        magInt = bs.BitStream(bin=magBin).int
#                        print ini, leng, 'Magnitude AC magInt-magBin: ', magInt, magBin
                        if magInt == 0 and cat == 1:
                            magdif = -1
                        elif magInt == -1 and cat == 1:
                            magdif = 1
                        elif magInt < 0:
                            magdif = ( self.DiffCat[cat][1]+magInt )+1
                        else: # magInt < -1:
                            magdif = (-1*self.DiffCat[cat][1])+magInt
                            
                        seqaux.append(magdif)
#                        print 'SEQUENCIAux AC: ', seqaux, basecode + magBin #coefsq

                        ini = leng
                        if ch == 0: leng = ini + 4
                        else:       leng += 2
                        basecode = seqhuff[ini:leng]
                        value = (leng-ini, basecode)
                        
                    if (basecode == '1010') or (basecode == '00'):
                        break
                        
                nblocks += 1
                ini = leng
                leng = ini + 2
                basecode = seqhuff[ini:leng]
                value = (leng-ini, basecode)
#                print "Reinit Cont: ini-leng ", ini, leng, basecode
                coefsq.append(seqaux)
                seqaux = []
        
        return (nblocks, coefsq)
    
    def coefcat(self, mag):
        '''
        Encontra a categoria da magnitude do coeficiente.
        
        input: mag -> int
        output: cat -> int
        '''
        difcat = self.DiffCat
#        difcat = [ (0), (1),(2,3), (4,7), (8,15), (16,31), (32,63), (64,127), (128,255), (256,511),
#                   (512,1023), (1024,2047),(2048,4095), (4096,8191), (8192,16383), (16384,32767) ]
        
        if mag < 0: mag = abs(mag)
        if mag > 32767:
            mag=32767
            return 15
        if mag==0:
            return 0
        elif mag==-1 or mag==1:
            return 1
            
        for cat in range(2,len(difcat)):
            if difcat[cat][0]<=mag<=difcat[cat][1]:
                return cat

#    def acdctables(self):
#        '''
#        Huffman code Tables for AC and DC coefficient differences.
#        
#        output: (dcLumaTB, dcChroTB, acLumaTB, acChrmTB)
#        '''
#        dcLumaTB = { 0:(2,'00'),     1:(3,'010'),      2:(3,'011'),       3:(3,'100'),
#                     4:(3,'101'),    5:(3,'110'),      6:(4,'1110'),      7:(5,'11110'),
#                     8:(6,'111110'), 9:(7,'1111110'), 10:(8,'11111110'), 11:(9,'111111110')}
#    
#        dcChroTB = { 0:(2,'00'),       1:(2,'01'),         2:( 2,'10'),          3:( 3,'110'),
#                     4:(4,'1110'),     5:(5,'11110'),      6:( 6,'111110'),      7:( 7,'1111110'),
#                     8:(8,'11111110'), 9:(9,'111111110'), 10:(10,'1111111110'), 11:(11,'11111111110')}
#                     
#    #Table for luminance DC coefficient differences
#        #       [(run,category) : (size, 'codeword')]
#        acLumaTB = {( 0, 0):( 4,'1010'), #EOB
#                    ( 0, 1):( 2,'00'),               ( 0, 2):( 2,'01'),
#                    ( 0, 3):( 3,'100'),              ( 0, 4):( 4,'1011'),
#                    ( 0, 5):( 5,'11010'),            ( 0, 6):( 7,'1111000'),
#                    ( 0, 7):( 8,'11111000'),         ( 0, 8):(10,'1111110110'),
#                    ( 0, 9):(16,'1111111110000010'), ( 0,10):(16,'1111111110000011'),
#                    ( 1, 1):( 4,'1100'),             ( 1, 2):( 5,'11011'),
#                    ( 1, 3):( 7,'1111001'),          ( 1, 4):( 9,'111110110'),
#                    ( 1, 5):(11,'11111110110'),      ( 1, 6):(16,'1111111110000100'),
#                    ( 1, 7):(16,'1111111110000101'), ( 1, 8):(16,'1111111110000110'),
#                    ( 1, 9):(16,'1111111110000111'), ( 1,10):(16,'1111111110001000'),
#                    ( 2, 1):( 5,'11100'),            ( 2, 2):( 8,'11111001'),
#                    ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110100'),
#                    ( 2, 5):(16,'1111111110001001'), ( 2, 6):(16,'1111111110001010'),
#                    ( 2, 7):(16,'1111111110001011'), ( 2, 8):(16,'1111111110001100'),
#                    ( 2, 9):(16,'1111111110001101'), ( 2,10):(16,'1111111110001110'),
#                    ( 3, 1):( 6,'111010'),           ( 3, 2):( 9,'111110111'),
#                    ( 3, 3):(12,'111111110101'),     ( 3, 4):(16,'1111111110001111'),
#                    ( 3, 5):(16,'1111111110010000'), ( 3, 6):(16,'1111111110010001'),
#                    ( 3, 7):(16,'1111111110010010'), ( 3, 8):(16,'1111111110010011'),
#                    ( 3, 9):(16,'1111111110010100'), ( 3,10):(16,'1111111110010101'),
#                    ( 4, 1):( 6,'111011'),           ( 4, 2):(10,'1111111000'),
#                    ( 4, 3):(16,'1111111110010110'), ( 4, 4):(16,'1111111110010111'),
#                    ( 4, 5):(16,'1111111110011000'), ( 4, 6):(16,'1111111110011001'),
#                    ( 4, 7):(16,'1111111110011010'), ( 4, 8):(16,'1111111110011011'),
#                    ( 4, 9):(16,'1111111110011100'), ( 4,10):(16,'1111111110011101'),
#                    ( 5, 1):( 7,'1111010'),          ( 5, 2):(11,'11111110111'),
#                    ( 5, 3):(16,'1111111110011110'), ( 5, 4):(16,'1111111110011111'),
#                    ( 5, 5):(16,'1111111110100000'), ( 5, 6):(16,'1111111110100001'),
#                    ( 5, 7):(16,'1111111110100010'), ( 5, 8):(16,'1111111110100011'),
#                    ( 5, 9):(16,'1111111110100100'), ( 5,10):(16,'1111111110100101'),
#                    ( 6, 1):( 7,'1111011'),          ( 6, 2):(12,'111111110110'),
#                    ( 6, 3):(16,'1111111110100110'), ( 6, 4):(16,'1111111110100111'),
#                    ( 6, 5):(16,'1111111110101000'), ( 6, 6):(16,'1111111110101001'),
#                    ( 6, 7):(16,'1111111110101010'), ( 6, 8):(16,'1111111110101011'),
#                    ( 6, 9):(16,'1111111110101100'), ( 6,10):(16,'1111111110101101'),
#                    ( 7, 1):( 8,'11111010'),         ( 7, 2):(12,'111111110111'),
#                    ( 7, 3):(16,'1111111110101110'), ( 7, 4):(16,'1111111110101111'),
#                    ( 7, 5):(16,'1111111110110000'), ( 7, 6):(16,'1111111110110001'),
#                    ( 7, 7):(16,'1111111110110010'), ( 7, 8):(16,'1111111110110011'),
#                    ( 7, 9):(16,'1111111110110100'), ( 7,10):(16,'1111111110110101'),
#                    ( 8, 1):( 9,'111111000'),        ( 8, 2):(15,'111111111000000'),
#                    ( 8, 3):(16,'1111111110110110'), ( 8, 4):(16,'1111111110110111'),
#                    ( 8, 5):(16,'1111111110111000'), ( 8, 6):(16,'1111111110111001'),
#                    ( 8, 7):(16,'1111111110111010'), ( 8, 8):(16,'1111111110111011'),
#                    ( 8, 9):(16,'1111111110111100'), ( 8,10):(16,'1111111110111101'),
#                    ( 9, 1):( 9,'111111001'),        ( 9, 2):(16,'1111111110111110'),
#                    ( 9, 3):(16,'1111111110111111'), ( 9, 4):(16,'1111111111000000'),
#                    ( 9, 5):(16,'1111111111000001'), ( 9, 6):(16,'1111111111000010'),
#                    ( 9, 7):(16,'1111111111000011'), ( 9, 8):(16,'1111111111000100'),
#                    ( 9, 9):(16,'1111111111000101'), ( 9,10):(16,'1111111111000110'),
#                    (10, 1):( 9,'111111010'),        (10, 2):(16,'1111111111000111'),
#                    (10, 3):(16,'1111111111001000'), (10, 4):(16,'1111111111001001'),
#                    (10, 5):(16,'1111111111001010'), (10, 6):(16,'1111111111001011'),
#                    (10, 7):(16,'1111111111001100'), (10, 8):(16,'1111111111001101'),
#                    (10, 9):(16,'1111111111001110'), (10,10):(16,'1111111111001111'),
#                    (11, 1):(10,'1111111001'),       (11, 2):(16,'1111111111010000'),
#                    (11, 3):(16,'1111111111010001'), (11, 4):(16,'1111111111010010'),
#                    (11, 5):(16,'1111111111010011'), (11, 6):(16,'1111111111010100'),
#                    (11, 7):(16,'1111111111010101'), (11, 8):(16,'1111111111010110'),
#                    (11, 9):(16,'1111111111010111'), (11,10):(16,'1111111111011000'),
#                    (12, 1):(10,'1111111010'),       (12, 2):(16,'1111111111011001'),
#                    (12, 3):(16,'1111111111011010'), (12, 4):(16,'1111111111011011'),
#                    (12, 5):(16,'1111111111011100'), (12, 6):(16,'1111111111011101'),
#                    (12, 7):(16,'1111111111011110'), (12, 8):(16,'1111111111011111'),
#                    (12, 9):(16,'1111111111100000'), (12,10):(16,'1111111111100001'),
#                    (13, 1):(11,'11111111000'),      (13, 2):(16,'1111111111100010'),
#                    (13, 3):(16,'1111111111100011'), (13, 4):(16,'1111111111100100'),
#                    (13, 5):(16,'1111111111100101'), (13, 6):(16,'1111111111100110'),
#                    (13, 7):(16,'1111111111100111'), (13, 8):(16,'1111111111101000'),
#                    (13, 9):(16,'1111111111101001'), (13,10):(16,'1111111111101010'),
#                    (14, 1):(16,'1111111111101011'), (14, 2):(16,'1111111111101100'),
#                    (14, 3):(16,'1111111111101101'), (14, 4):(16,'1111111111101110'),
#                    (14, 5):(16,'1111111111101111'), (14, 6):(16,'1111111111110000'),
#                    (14, 7):(16,'1111111111110001'), (14, 8):(16,'1111111111110010'),
#                    (14, 9):(16,'1111111111110011'), (14,10):(16,'1111111111110100'),
#                    (15, 0):(11,'11111111001'),     #(ZRL)
#                    (15, 1):(16,'1111111111110101'), (15, 2):(16,'1111111111110110'),
#                    (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
#                    (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
#                    (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
#                    (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
#                
#    #Table for chrominance AC coefficients
#        acChrmTB = {( 0, 0):( 2,'00'), #EOB
#                    ( 0, 1):( 2,'01'),               ( 0, 2):( 3,'100'),
#                    ( 0, 3):( 4,'1010'),             ( 0, 4):( 5,'11000'),
#                    ( 0, 5):( 5,'11001'),            ( 0, 6):( 6,'111000'),
#                    ( 0, 7):( 7,'1111000'),          ( 0, 8):( 9,'111110100'),
#                    ( 0, 9):(10,'1111110110'),       ( 0,10):(12,'111111110100'),
#                    ( 1, 1):( 4,'1011'),             ( 1, 2):( 6,'111001'),
#                    ( 1, 3):( 8,'11110110'),         ( 1, 4):( 9,'111110101'),
#                    ( 1, 5):(11,'11111110110'),      ( 1, 6):(12,'111111110101'),
#                    ( 1, 7):(16,'1111111110001000'), ( 1, 8):(16,'1111111110001001'),
#                    ( 1, 9):(16,'1111111110001010'), ( 1,10):(16,'1111111110001011'),
#                    ( 2, 1):( 5,'11010'),            ( 2, 2):( 8,'11110111'),
#                    ( 2, 3):(10,'1111110111'),       ( 2, 4):(12,'111111110110'),
#                    ( 2, 5):(15,'111111111000010'),  ( 2, 6):(16,'1111111110001100'),
#                    ( 2, 7):(16,'1111111110001101'), ( 2, 8):(16,'1111111110001110'),
#                    ( 2, 9):(16,'1111111110001111'), ( 2,10):(16,'1111111110010000'),
#                    ( 3, 1):( 5,'11011'),            ( 3, 2):( 8,'11111000'),
#                    ( 3, 3):(10,'1111111000'),       ( 3, 4):(12,'111111110111'),
#                    ( 3, 5):(16,'1111111110010001'), ( 3, 6):(16,'1111111110010010'),
#                    ( 3, 7):(16,'1111111110010011'), ( 3, 8):(16,'1111111110010100'),
#                    ( 3, 9):(16,'1111111110010101'), ( 3,10):(16,'1111111110010110'),
#                    ( 4, 1):( 6,'111010'),           ( 4, 2):( 9,'111110110'),
#                    ( 4, 3):(16,'1111111110010111'), ( 4, 4):(16,'1111111110011000'),
#                    ( 4, 5):(16,'1111111110011001'), ( 4, 6):(16,'1111111110011010'),
#                    ( 4, 7):(16,'1111111110011011'), ( 4, 8):(16,'1111111110011100'),
#                    ( 4, 9):(16,'1111111110011101'), ( 4,10):(16,'1111111110011110'),
#                    ( 5, 1):( 6,'111011'),           ( 5, 2):(10,'1111111001'),
#                    ( 5, 3):(16,'1111111110011111'), ( 5, 4):(16,'1111111110100000'),
#                    ( 5, 5):(16,'1111111110100001'), ( 5, 6):(16,'1111111110100010'),
#                    ( 5, 7):(16,'1111111110100011'), ( 5, 8):(16,'1111111110100100'),
#                    ( 5, 9):(16,'1111111110100101'), ( 5,10):(16,'1111111110100110'),
#                    ( 6, 1):( 7,'1111001'),          ( 6, 2):(11,'11111110111'),
#                    ( 6, 3):(16,'1111111110100111'), ( 6, 4):(16,'1111111110101000'),
#                    ( 6, 5):(16,'1111111110101001'), ( 6, 6):(16,'1111111110101010'),
#                    ( 6, 7):(16,'1111111110101011'), ( 6, 8):(16,'1111111110101100'),
#                    ( 6, 9):(16,'1111111110101101'), ( 6,10):(16,'1111111110101110'),
#                    ( 7, 1):( 7,'1111010'),          ( 7, 2):(11,'11111111000'),
#                    ( 7, 3):(16,'1111111110101111'), ( 7, 4):(16,'1111111110110000'),
#                    ( 7, 5):(16,'1111111110110001'), ( 7, 6):(16,'1111111110110010'),
#                    ( 7, 7):(16,'1111111110110011'), ( 7, 8):(16,'1111111110110100'),
#                    ( 7, 9):(16,'1111111110110101'), ( 7,10):(16,'1111111110110110'),
#                    ( 8, 1):( 8,'11111001'),         ( 8, 2):(16,'1111111110110111'),
#                    ( 8, 3):(16,'1111111110111000'), ( 8, 4):(16,'1111111110111001'),
#                    ( 8, 5):(16,'1111111110111010'), ( 8, 6):(16,'1111111110111011'),
#                    ( 8, 7):(16,'1111111110111100'), ( 8, 8):(16,'1111111110111101'),
#                    ( 8, 9):(16,'1111111110111110'), ( 8,10):(16,'1111111110111111'),
#                    ( 9, 1):( 9,'111110111'),        ( 9, 2):(16,'1111111111000000'),
#                    ( 9, 3):(16,'1111111111000001'), ( 9, 4):(16,'1111111111000010'),
#                    ( 9, 5):(16,'1111111111000011'), ( 9, 6):(16,'1111111111000100'),
#                    ( 9, 7):(16,'1111111111000101'), ( 9, 8):(16,'1111111111000110'),
#                    ( 9, 9):(16,'1111111111000111'), ( 9,10):(16,'1111111111001000'),
#                    (10, 1):( 9,'111111000'),        (10, 2):(16,'1111111111001001'),
#                    (10, 3):(16,'1111111111001010'), (10, 4):(16,'1111111111001011'),
#                    (10, 5):(16,'1111111111001100'), (10, 6):(16,'1111111111001101'),
#                    (10, 7):(16,'1111111111001110'), (10, 8):(16,'1111111111001111'),
#                    (10, 9):(16,'1111111111010000'), (10,10):(16,'1111111111010001'),
#                    (11, 1):( 9,'111111001'),        (11, 2):(16,'1111111111010010'),
#                    (11, 3):(16,'1111111111010011'), (11, 4):(16,'1111111111010100'),
#                    (11, 5):(16,'1111111111010101'), (11, 6):(16,'1111111111010110'),
#                    (11, 7):(16,'1111111111010111'), (11, 8):(16,'1111111111011000'),
#                    (11, 9):(16,'1111111111011001'), (11,10):(16,'1111111111011010'),
#                    (12, 1):( 9,'111111010'),        (12, 2):(16,'1111111111011011'),
#                    (12, 3):(16,'1111111111011100'), (12, 4):(16,'1111111111011101'),
#                    (12, 5):(16,'1111111111011110'), (12, 6):(16,'1111111111011111'),
#                    (12, 7):(16,'1111111111100000'), (12, 8):(16,'1111111111100001'),
#                    (12, 9):(16,'1111111111100010'), (12,10):(16,'1111111111100011'),
#                    (13, 1):(11,'11111111001'),      (13, 2):(16,'1111111111100100'),
#                    (13, 3):(16,'1111111111100101'), (13, 4):(16,'1111111111100110'),
#                    (13, 5):(16,'1111111111100111'), (13, 6):(16,'1111111111101000'),
#                    (13, 7):(16,'1111111111101001'), (13, 8):(16,'1111111111101010'),
#                    (13, 9):(16,'1111111111101011'), (13,10):(16,'1111111111101100'),
#                    (14, 1):(14,'11111111100000'),   (14, 2):(16,'1111111111101101'),
#                    (14, 3):(16,'1111111111101110'), (14, 4):(16,'1111111111101111'),
#                    (14, 5):(16,'1111111111110000'), (14, 6):(16,'1111111111110001'),
#                    (14, 7):(16,'1111111111110010'), (14, 8):(16,'1111111111110011'),
#                    (14, 9):(16,'1111111111110100'), (14,10):(16,'1111111111110101'),
#                    (15, 0):(10,'1111111010'),       #(ZRL)
#                    (15, 1):(15,'111111111000011'),  (15, 2):(16,'1111111111110110'),
#                    (15, 3):(16,'1111111111110111'), (15, 4):(16,'1111111111111000'),
#                    (15, 5):(16,'1111111111111001'), (15, 6):(16,'1111111111111010'),
#                    (15, 7):(16,'1111111111111011'), (15, 8):(16,'1111111111111100'),
#                    (15, 9):(16,'1111111111111101'), (15,10):(16,'1111111111111110')}
#                    
#        return (dcLumaTB, dcChroTB, acLumaTB, acChrmTB)
#FIM CLASS HUFFCODER

def adjImg(img, blocksize=[8,8]):
    '''
    Ajusta as dimensões da imagem de modo a serem multiplas de 'blocksize'.
    blocksize=[m, n]
    '''
    
    Mo, No, Do = img.shape
    
    if int(Mo)%blocksize[0] != 0:
        M = Mo + (blocksize[0] - int(Mo)%blocksize[0])
    else:
        M = Mo
        
    if int(No)%blocksize[1] != 0:
        N = No + (blocksize[1] - int(No)%blocksize[1])
    else:
        N = No
        
    newImg = np.zeros((M,N,Do),np.float32)
    newImg[:Mo,:No] = img
        
    return (M, N, Do), newImg
    
#def genQntb(qualy):
#    '''
#    Gera tabela para quantização.
#    
#    input: qualy -> int [1-99]
#    '''
#    
#    fact = qualy
#    Z = np.array([[[16., 17., 17.], [11., 18., 18.], [10., 24., 24.], [16., 47., 47.], [124., 99., 99.], [140., 99., 99.], [151., 99., 99.], [161., 99., 99.]],
#                  [[12., 18., 18.], [12., 21., 21.], [14., 26., 26.], [19., 66., 66.], [ 26., 99., 99.], [158., 99., 99.], [160., 99., 99.], [155., 99., 99.]],
#                  [[14., 24., 24.], [13., 26., 26.], [16., 56., 56.], [24., 99., 99.], [ 40., 99., 99.], [157., 99., 99.], [169., 99., 99.], [156., 99., 99.]],
#                  [[14., 47., 47.], [17., 66., 66.], [22., 99., 99.], [29., 99., 99.], [ 51., 99., 99.], [187., 99., 99.], [180., 99., 99.], [162., 99., 99.]],
#                  [[18., 99., 99.], [22., 99., 99.], [37., 99., 99.], [56., 99., 99.], [ 68., 99., 99.], [109., 99., 99.], [103., 99., 99.], [177., 99., 99.]],
#                  [[24., 99., 99.], [35., 99., 99.], [55., 99., 99.], [64., 99., 99.], [ 81., 99., 99.], [104., 99., 99.], [113., 99., 99.], [192., 99., 99.]],
#                  [[49., 99., 99.], [64., 99., 99.], [78., 99., 99.], [87., 99., 99.], [103., 99., 99.], [121., 99., 99.], [120., 99., 99.], [101., 99., 99.]],
#                  [[72., 99., 99.], [92., 99., 99.], [95., 99., 99.], [98., 99., 99.], [112., 99., 99.], [100., 99., 99.], [103., 99., 99.], [199., 99., 99.]]])
#                  
#    if qualy < 1 : fact = 1
#    if qualy > 99: fact = 99
#    if qualy < 50:
#        qualy = 5000 / fact
#    else:
#        qualy = 200 - 2*fact
#        
#    qZ = ((Z*qualy) + 50)/100
#    qZ[qZ<1] = 1
#    qZ[qZ>255] = 255
#    
#    return qZ

def zigzag(coefs, shp=[8,8]):
    '''
    Retorna a sequência de coeficiêntes ordenados de acordo com o padrão Zigue-Zague
    '''
    
    coefseq=[]
    
    indx = sorted(((x,y) for x in range(shp[0]) for y in range(shp[1])),
                        key = lambda (x,y): (x+y, -y if (x+y) % 2 else y))
                        
    for ind in range(len(indx)):
        coefseq.append( coefs[indx[ind]] )
                
    nelmnt = len(coefseq)   #Numero de elementos em 'coefseq' (ordenados em zig-zag)
    seq1D = []               
    i=-1; nz=0
    while abs(i)<=(nelmnt):#len(coefseq):
        if coefseq[i]==0.0:
            i-=1
            nz += 1
        else:
            seq1D = coefseq[0:(nelmnt-abs(i))+1]
            break
            
    if abs(i)>nelmnt:
        seq1D = [0]
    return seq1D     
                
def zagzig(seq, bshp=[8,8]): #imshape, 
    '''
    '''
    
    block = np.zeros(bshp)
    #seqsz = len(seq1d)
    #M, N = imshape
        
    indx = sorted(((x,y) for x in range(bshp[0]) for y in range(bshp[1])),
                        key = lambda (x,y): (x+y, -y if (x+y) % 2 else y))
    if len(seq)>0:                    
        for s in range(len(seq)): #        for t in range(len(seq[s])):
            block[indx[s]] = seq[s]
        
    return np.float_(block)
        
#def dct3ch(subimg):
#    '''
#    Aplica dct para cada um dos canais da subimagem.
#    '''
#    
#    coefs = np.zeros(subimg.shape)
#    
#    for ch in range(3):
#        coefs[:,:,ch] = cv2.dct(subimg[:,:,ch])
#        
#    return coefs