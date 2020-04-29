

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import zemberek.core.logging.Log;
import zemberek.morphology.analysis.AnalysisFormatters;
import zemberek.morphology.analysis.InformalAnalysisConverter;
import zemberek.morphology.analysis.SingleAnalysis;
import zemberek.morphology.TurkishMorphology;
import zemberek.morphology.analysis.WordAnalysis;
import zemberek.morphology.generator.WordGenerator.Result;
import zemberek.morphology.lexicon.DictionaryItem;
import zemberek.morphology.lexicon.RootLexicon;
import zemberek.normalization.TurkishSpellChecker;


public class Main {

    public static void main(String[] args) throws IOException {


        TurkishMorphology morphology2 = TurkishMorphology.createWithDefaults();
        TurkishSpellChecker spellChecker = new TurkishSpellChecker(morphology2);

        // The name of the file to open.
        String fileName = "normalizingDatas.txt"; //okunacak dosya
        String fileName2 = "normalizedDatas.txt"; //yaz�lacak dosya


        try {
            File file = new File(fileName);
            File file2 = new File(fileName2);


            BufferedReader read = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF8"));   // turkce char problemi icin
            Writer write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file2), "UTF8"));           // utf-8


            String str;
            int i = 0;
            while ((str = read.readLine()) != null) {   // okuma islemi
                str = str.substring(str.indexOf(',') + 1, str.length());  // virgul ve oncesini at�yoruz
                String[] words = str.split(" "); // kelimeleri ayirdik
                String pivot = "";
                String newSentence = "";  // hatali yazildigini dusundugu kelimeleri yenileriyle degistirecegimiz yeni cumle degiskeni

                for (int j = 0; j < words.length; j++) {

                    if (spellChecker.check(words[j]))
                        pivot = words[j];
                    else {
                        if (!spellChecker.suggestForWord(words[j]).isEmpty())  // eger hatali yazilmis ise spellChecker'dan yeni kelime istiyoruz
                            pivot = spellChecker.suggestForWord(words[j]).get(0); // ilk tahminini cekmeyi deniyoruz
                        else
                            pivot = words[j];
                    }


                    if (j == 0)
                        newSentence = pivot;
                    else
                        newSentence = newSentence + " " + pivot;


                }

                write.append(newSentence).append("\r\n");

            }

            read.close();
            write.flush();
            write.close();


        } catch (UnsupportedEncodingException e) {
            System.out.println(e.getMessage());
        } catch (IOException e) {
            System.out.println(e.getMessage());
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }


    }

}