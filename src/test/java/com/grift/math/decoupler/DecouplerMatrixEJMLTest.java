package com.grift.math.decoupler;

import com.grift.forex.symbol.SymbolIndexMap;
import org.jetbrains.annotations.NotNull;

public class DecouplerMatrixEJMLTest extends DecouplerMatrixTest {
    @NotNull
    @Override
    public Factory getFactory(SymbolIndexMap symbolIndexMap) {
        return new DecouplerMatrixEJMLImpl.EJMLFactory(symbolIndexMap);
    }
}
