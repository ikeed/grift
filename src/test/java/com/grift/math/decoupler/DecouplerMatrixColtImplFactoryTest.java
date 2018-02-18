package com.grift.math.decoupler;

import com.grift.forex.symbol.SymbolIndexMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class DecouplerMatrixColtImplFactoryTest {
    private SymbolIndexMap symbolIndexMap;
    private DecouplerMatrix.Factory factory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
        symbolIndexMap.addSymbol("CAD");
        symbolIndexMap.addSymbol("USD");
        symbolIndexMap.addSymbol("GBP");
        factory = new DecouplerMatrix.Factory(symbolIndexMap);
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

        DecouplerMatrixColtImpl mat = factory.make(vals);

        assertEquals(symbolIndexMap.size(), mat.rows());
        for (int i = 0; i < symbolIndexMap.size(); i++) {
            for (int j = 0; j < symbolIndexMap.size(); j++) {
                assertEquals(String.format("(i, j) = (%d, %d)", i, j), vals[i][j], mat.get(i, j), 0.000001d);
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