package com.grift.math.decoupler;

import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertNotNull;

@RunWith(MockitoJUnitRunner.class)
public class DecouplerMatrixColtImplTest extends DecouplerMatrixTest {
    @NotNull
    @Override
    protected Factory getFactory() {
        return new DecouplerMatrixColtImpl.Factory(symbolIndexMap);
    }

    @Test
    public void getFactoryTest() {
        assertNotNull(getFactory());
    }
}
