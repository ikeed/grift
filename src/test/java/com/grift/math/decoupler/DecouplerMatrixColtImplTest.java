package com.grift.math.decoupler;

import org.jetbrains.annotations.NotNull;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class DecouplerMatrixColtImplTest extends DecouplerMatrixTest {
    @NotNull
    @Override
    protected DecouplerMatrix.Factory getFactory() {
        return new DecouplerMatrix.Factory(symbolIndexMap);
    }
}
