/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef AIV_UDMA_COMM_H
#define AIV_UDMA_COMM_H

#include "kernel_operator.h"
#include "moe_distribute_base.h"

namespace DeepEpAivUdma {

constexpr uint64_t HNS_ROCE_OWNER_SHIFT = 15UL;
constexpr uint64_t HNS_ROCE_DB_PI_MOD = 65536UL;
constexpr uint64_t AIV_UDMA_UINT32_MAX = 0xFFFFFFFFULL;
constexpr uint32_t HNS_ROCE_RDMA_WRITE_OPCODE = 3U;
constexpr uint32_t HNS_ROCE_SEND_SIGNALED = 1U << 8U;
constexpr uint32_t HNS_ROCE_NUM_SGE_ONE = 1U << 24U;
constexpr uint64_t HNS_ROCE_SQ_DB_CMD = 0UL;
constexpr uint64_t HNS_ROCE_SQ_DB_CMD_SHIFT = 24UL;
constexpr uint64_t HNS_ROCE_SQ_DB_PI_SHIFT = 32UL;
constexpr uint64_t HNS_ROCE_SQ_DB_SL_SHIFT = 48UL;

__aicore__ inline bool IsValidMemType(HcclAiRMAMemType memType)
{
    return static_cast<uint32_t>(memType) < static_cast<uint32_t>(HcclAiRMAMemType::MAX_NUM);
}

struct AivUdmaWriteParams {
    GM_ADDR srcAddr{0};
    GM_ADDR dstAddr{0};
    uint64_t length{0};
    uint64_t dstRankId{0};
    uint32_t qpIndex{0};
    HcclAiRMAMemType remoteMemType{HcclAiRMAMemType::REMOTE_INPUT};
    HcclAiRMAMemType localMemType{HcclAiRMAMemType::LOCAL_OUTPUT};
    bool signaled{true};
};

__aicore__ inline __gm__ HcclAiRMAInfo *GetAiRMAInfoFromA2Context(GM_ADDR contextGM)
{
    if (contextGM == 0U) {
        return nullptr;
    }
    __gm__ HcclA2CombineOpParam *context = (__gm__ HcclA2CombineOpParam *)contextGM;
    if (context->aiRMAInfo == 0U || context->sizeOfAiRMAInfo < sizeof(HcclAiRMAInfo)) {
        return nullptr;
    }
    return (__gm__ HcclAiRMAInfo *)(context->aiRMAInfo);
}

__aicore__ inline GM_ADDR GetWindowAddr(__gm__ HcclOpParam *winContext, uint32_t rankId, uint32_t curRankId,
                                        uint64_t offset = 0)
{
    return GetBaseWindAddrByRankId(winContext, rankId, curRankId) + offset;
}

__aicore__ inline GM_ADDR GetWindowStateAddr(__gm__ HcclOpParam *winContext, uint32_t rankId, uint32_t curRankId,
                                             uint64_t offset = 0)
{
    return GetBaseWindStateAddrByRankId(winContext, rankId, curRankId) + offset;
}

__aicore__ inline __gm__ HcclAiRMAWQ *GetSendQueue(__gm__ HcclAiRMAInfo *qpInfo, uint64_t dstRankId,
                                                   uint32_t qpIndex = 0)
{
    if (qpInfo == nullptr || qpInfo->sqPtr == 0U || qpInfo->rankNum == 0U || qpInfo->qpNum == 0U ||
        qpInfo->sizeOfAiRMAWQ == 0U || dstRankId >= qpInfo->rankNum || qpIndex >= qpInfo->qpNum) {
        return nullptr;
    }
    uint64_t qpNum = qpInfo->qpNum;
    return (__gm__ HcclAiRMAWQ *)(qpInfo->sqPtr + (dstRankId * qpNum + qpIndex) *
                                                       static_cast<uint64_t>(qpInfo->sizeOfAiRMAWQ));
}

__aicore__ inline __gm__ HcclAiRMACQ *GetSendCompletionQueue(__gm__ HcclAiRMAInfo *qpInfo, uint64_t dstRankId,
                                                             uint32_t qpIndex = 0)
{
    if (qpInfo == nullptr || qpInfo->scqPtr == 0U || qpInfo->rankNum == 0U || qpInfo->qpNum == 0U ||
        qpInfo->sizeOfAiRMACQ == 0U || dstRankId >= qpInfo->rankNum || qpIndex >= qpInfo->qpNum) {
        return nullptr;
    }
    uint64_t qpNum = qpInfo->qpNum;
    return (__gm__ HcclAiRMACQ *)(qpInfo->scqPtr + (dstRankId * qpNum + qpIndex) *
                                                        static_cast<uint64_t>(qpInfo->sizeOfAiRMACQ));
}

__aicore__ inline __gm__ MemDetails *GetMemDetail(__gm__ HcclAiRMAInfo *qpInfo, uint64_t rankId,
                                                  HcclAiRMAMemType memType)
{
    if (qpInfo == nullptr || rankId >= qpInfo->rankNum || !IsValidMemType(memType) || qpInfo->memPtr == 0U ||
        qpInfo->sizeOfAiRMAMem == 0U) {
        return nullptr;
    }
    __gm__ HcclAiRMAMemInfo *memInfo =
        (__gm__ HcclAiRMAMemInfo *)(qpInfo->memPtr + rankId * static_cast<uint64_t>(qpInfo->sizeOfAiRMAMem));
    if (memInfo->memDetailPtr == 0U || memInfo->sizeOfMemDetails == 0U ||
        static_cast<uint32_t>(memType) >= memInfo->memMaxNum) {
        return nullptr;
    }
    return (__gm__ MemDetails *)(memInfo->memDetailPtr +
                                 memInfo->sizeOfMemDetails * static_cast<uint32_t>(memType));
}

__aicore__ inline bool WaitSqSpace(__gm__ HcclAiRMAWQ *sendQueue, uint32_t curHead, uint32_t maxPollCount = 0)
{
    if (sendQueue == nullptr || sendQueue->tailAddr == 0U || sendQueue->depth <= 1U) {
        return false;
    }
    uint32_t pollCount = 0;
    while (true) {
        cacheWriteThrough((__gm__ uint8_t *)sendQueue->tailAddr, sizeof(uint32_t));
        uint32_t curTail = *(__gm__ uint32_t *)(sendQueue->tailAddr);
        if ((curHead - curTail) < sendQueue->depth - 1U) {
            return true;
        }
        if (maxPollCount != 0U && ++pollCount >= maxPollCount) {
            return false;
        }
        int64_t unusedCycle = AscendC::GetSystemCycle();
        (void)unusedCycle;
    }
}

__aicore__ inline bool FillRdmaWriteWqe(__gm__ uint8_t *wqeAddr, __gm__ HcclAiRMAInfo *qpInfo,
                                        __gm__ HcclAiRMAWQ *sendQueue, const AivUdmaWriteParams &params,
                                        uint32_t curHead)
{
    if (wqeAddr == nullptr) {
        return false;
    }
    uint64_t ownBit = (static_cast<uint64_t>(curHead) >> HNS_ROCE_OWNER_SHIFT) & 1UL;
    uint32_t byte4 = HNS_ROCE_RDMA_WRITE_OPCODE;
    byte4 |= ((~ownBit) << 7U) & (1U << 7U);
    if (params.signaled) {
        byte4 |= HNS_ROCE_SEND_SIGNALED;
    }

    __gm__ MemDetails *remoteMem = GetMemDetail(qpInfo, params.dstRankId, params.remoteMemType);
    __gm__ MemDetails *localMem = GetMemDetail(qpInfo, params.dstRankId, params.localMemType);
    if (remoteMem == nullptr || localMem == nullptr) {
        return false;
    }

    *(__gm__ uint32_t *)(wqeAddr) = byte4;
    *(__gm__ uint32_t *)(wqeAddr + 4) = static_cast<uint32_t>(params.length);
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0U;
    *(__gm__ uint32_t *)(wqeAddr + 12) = HNS_ROCE_NUM_SGE_ONE;
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0U;
    *(__gm__ uint32_t *)(wqeAddr + 20) = remoteMem->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = reinterpret_cast<uint64_t>(params.dstAddr);

    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = static_cast<uint32_t>(params.length);
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) = localMem->key;
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) = reinterpret_cast<uint64_t>(params.srcAddr);

    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    (void)sendQueue;
    return true;
}

__aicore__ inline uint64_t MakeSqDoorbell(__gm__ HcclAiRMAWQ *sendQueue, uint32_t nextHead)
{
    uint64_t doorbell = 0UL;
    doorbell |= static_cast<uint64_t>(sendQueue->wqn);
    doorbell |= HNS_ROCE_SQ_DB_CMD << HNS_ROCE_SQ_DB_CMD_SHIFT;
    doorbell |= (static_cast<uint64_t>(nextHead) % HNS_ROCE_DB_PI_MOD) << HNS_ROCE_SQ_DB_PI_SHIFT;
    doorbell |= static_cast<uint64_t>(sendQueue->sl) << HNS_ROCE_SQ_DB_SL_SHIFT;
    return doorbell;
}

__aicore__ inline void RingDoorbell(__gm__ HcclAiRMAWQ *sendQueue, uint32_t nextHead,
                                    AscendC::LocalTensor<uint64_t> &doorbellLocal,
                                    AscendC::LocalTensor<uint32_t> &headLocal)
{
    doorbellLocal.SetValue(0, MakeSqDoorbell(sendQueue, nextHead));
    AscendC::GlobalTensor<uint64_t> doorbellGlobal;
    doorbellGlobal.SetGlobalBuffer((__gm__ uint64_t *)(sendQueue->dbAddr));
    AscendC::DataCopyExtParams doorbellCopyParams{1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(doorbellGlobal, doorbellLocal, doorbellCopyParams);
    PipeBarrier<PIPE_ALL>();

    headLocal.SetValue(0, nextHead);
    AscendC::GlobalTensor<uint32_t> headGlobal;
    headGlobal.SetGlobalBuffer((__gm__ uint32_t *)(sendQueue->headAddr));
    AscendC::DataCopyExtParams headCopyParams{1U, static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(headGlobal, headLocal, headCopyParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool PostRdmaWrite(__gm__ HcclAiRMAInfo *qpInfo, const AivUdmaWriteParams &params,
                                     AscendC::LocalTensor<uint64_t> &doorbellLocal,
                                     AscendC::LocalTensor<uint32_t> &headLocal, uint32_t maxPollCount = 0)
{
    if (qpInfo == nullptr || qpInfo->rankNum == 0U || qpInfo->qpNum == 0U || qpInfo->sqPtr == 0U ||
        qpInfo->memPtr == 0U || qpInfo->sizeOfAiRMAWQ == 0U || qpInfo->sizeOfAiRMAMem == 0U ||
        params.srcAddr == 0U || params.dstAddr == 0U || params.length == 0U ||
        params.length > AIV_UDMA_UINT32_MAX || !IsValidMemType(params.remoteMemType) ||
        !IsValidMemType(params.localMemType) ||
        params.dstRankId >= qpInfo->rankNum || params.qpIndex >= qpInfo->qpNum) {
        return false;
    }

    __gm__ HcclAiRMAWQ *sendQueue = GetSendQueue(qpInfo, params.dstRankId, params.qpIndex);
    if (sendQueue == nullptr || sendQueue->bufAddr == 0U || sendQueue->headAddr == 0U || sendQueue->tailAddr == 0U ||
        sendQueue->dbAddr == 0U || sendQueue->depth <= 1U ||
        sendQueue->wqeSize < sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg)) {
        return false;
    }
    cacheWriteThrough((__gm__ uint8_t *)sendQueue->headAddr, sizeof(uint32_t));
    uint32_t curHead = *(__gm__ uint32_t *)(sendQueue->headAddr);
    if (!WaitSqSpace(sendQueue, curHead, maxPollCount)) {
        return false;
    }

    __gm__ uint8_t *wqeAddr =
        (__gm__ uint8_t *)(sendQueue->bufAddr + sendQueue->wqeSize * (curHead % sendQueue->depth));
    if (!FillRdmaWriteWqe(wqeAddr, qpInfo, sendQueue, params, curHead)) {
        return false;
    }
    PipeBarrier<PIPE_ALL>();

    uint32_t nextHead = curHead + 1U;
    RingDoorbell(sendQueue, nextHead, doorbellLocal, headLocal);
    return true;
}

}  // namespace DeepEpAivUdma

#endif  // AIV_UDMA_COMM_H
