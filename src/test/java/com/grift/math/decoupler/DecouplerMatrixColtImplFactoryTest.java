package com.grift.math.decoupler;

import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class DecouplerMatrixColtImplFactoryTest {
    private SymbolIndexMap symbolIndexMap;
    private Factory factory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
        symbolIndexMap.addSymbol("CAD");
        symbolIndexMap.addSymbol("USD");
        symbolIndexMap.addSymbol("GBP");
        factory = new DecouplerMatrixColtImpl.Factory(symbolIndexMap);
    }

    @Test
    public void basicContstuctor() {
        DecouplerMatrix mat = factory.make();
        assertEquals(symbolIndexMap.size(), mat.rows());
    }

    @Test
    public void valuesConstructor() {
        double[][] vals = new double[symbolIndexMap.size()][symbolIndexMap.size()];
        for (int i = 0; i < symbolIndexMap.size(); i++) {
            for (int j = 0; j < symbolIndexMap.size(); j++) {
                vals[i][j] = 31 * i + 5 * j;
            }
        }

        DecouplerMatrix mat = factory.make(vals);

        assertEquals(symbolIndexMap.size(), mat.rows());
        for (String sym1 : symbolIndexMap.keySet()) {
            for (String sym2 : symbolIndexMap.keySet()) {
                if (!sym1.equals(sym2)) {
                    SymbolPair symbolPair = new SymbolPair(sym1 + sym2);
                    Integer[] indeces = symbolIndexMap.getIndecesForSymbolPair(symbolPair);
                    assertEquals(String.format("(i, j) = (%d, %d)", indeces[0], indeces[1]), vals[indeces[0]][indeces[1]], mat.get(symbolPair), 0.000001d);
                }
            }
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void valuesConstructorTooFew() {
        final int size = symbolIndexMap.size() - 1;
        double[][] vals = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                vals[i][j] = 31 * i + 5 * j;
            }
        }
        factory.make(vals);
    }
}