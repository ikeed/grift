package com.grift;

import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.math.decoupler.Factory;
import com.grift.spring.controller.DecoupleController;
import com.grift.spring.service.DecoupleService;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.Assert.assertNotNull;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = GriftApplication.class)
public class GriftApplicationTests {

    @Autowired
    DecoupleService decoupleService;

    @Autowired
    Factory decouplerFactory;

    @Autowired
    ImmutableSymbolIndexMap symbolIndexMap;

    @Autowired
    DecoupleController decoupleController;

    @Test
    public void contextLoads() {
        assertNotNull(decoupleService);
        assertNotNull(decouplerFactory);
        assertNotNull(symbolIndexMap);
        assertNotNull(decoupleController);
    }

    @Test
    public void startUp() {
        GriftApplication.main(new String[] {});
    }

}
